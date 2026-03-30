# simulation.py
# --------------------------------------------------
# Changes vs previous version:
# 1. Testing runs BEFORE epidemic step (quarantine before spread)
# 2. test_buffer always appended for all strategies
# 3. Lookback suffix parsing removed (unused)
# 4. agent.update() receives test_buffer for informed prior
# 5. GammaPoissonMAB removed
#
# Vectorisation (Option 1):
# 6. G is never deep-copied — topology is shared read-only.
#    build_sim_structures() pre-computes the CSR adjacency, block array,
#    precomputed edge_index tensor and block_id tensor once per graph.
# 7. State lives in a NumPy int8 array (state_arr) throughout.
#    G.nodes[n]['state'] is only synced back immediately before GNN reads.
# 8. node_level_testing() works directly on state_arr.
# 9. Daily state aggregation via np.bincount (single O(N) pass).
#
# GlobalGNTS optimisation:
# 10. The full 6-tuple from build_sim_structures is passed as `precomputed`
#     to GlobalGNTS.__init__ so it can cache edge_index_t and block_id_t
#     without rebuilding them.
# --------------------------------------------------

import copy
from collections import deque
import numpy as np
import networkx as nx

import config
from network_epidemic import (
    S, E, I, A, Q, R, STATE_NAMES,
    build_sim_structures,
    initialize_epidemic,
    sync_state_to_graph,
    run_seiqr_step,
)
from gnts import LocalGNTS, GlobalGNTS
from strategies import (
    BetaBinomialMAB,
    uniform_allocation, random_allocation, proportional_allocation,
)

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def infer_num_blocks(block_arr: np.ndarray) -> int:
    valid = block_arr[block_arr >= 0]
    return int(valid.max()) + 1 if valid.size > 0 else 0


def get_kits_for_day(day: int, schedule) -> int:
    kits = 0
    for start_day, num in schedule:
        if day >= start_day:
            kits = num
    return kits


# --------------------------------------------------
# Node-level testing  (operates on state_arr directly)
# --------------------------------------------------

def node_level_testing(state_arr, block_arr, block_id, nodes, G, num_kits):
    """Test up to num_kits nodes in block_id.

    Priority: confirmed I first, their S/E/A neighbours second,
    remaining S/E/A nodes third.
    state_arr modified in-place (positives set to Q=4).
    """
    if num_kits <= 0:
        return {}, 0

    in_block = np.where(block_arr == block_id)[0]
    if in_block.size == 0:
        return {}, 0

    # Local node->idx map for this block's neighbourhood lookup
    node_to_idx = {nodes[idx]: idx for idx in in_block}

    # Tier 1: I nodes in block
    tier1     = [idx for idx in in_block if state_arr[idx] == I]
    tier1_set = set(tier1)

    # Tier 2: S/E/A neighbours of tier1 that are also in block
    tier2_set = set()
    for idx in tier1:
        for nbr in G.neighbors(nodes[idx]):
            nbr_idx = node_to_idx.get(nbr)
            if nbr_idx is not None and state_arr[nbr_idx] in (S, E, A):
                tier2_set.add(nbr_idx)

    # Tier 3: remaining S/E/A in block
    tier3 = [
        idx for idx in in_block
        if state_arr[idx] in (S, E, A)
        and idx not in tier1_set
        and idx not in tier2_set
    ]

    queue     = tier1 + list(tier2_set) + tier3
    positives = {}
    wasted    = 0
    tested    = 0

    for idx in queue:
        if tested >= num_kits:
            break
        st = state_arr[idx]
        if st in (E, I, A):
            name = STATE_NAMES[st]
            positives[name] = positives.get(name, 0) + 1
            state_arr[idx]  = Q
        else:
            wasted += 1
        tested += 1

    return positives, wasted


# --------------------------------------------------
# Main simulation routine
# --------------------------------------------------

def run_simulation(strategy_name, G_template, sim_structures=None,
                   pretrained_gnts=None, kits_schedule=None):
    """Run one simulation episode.

    Parameters
    ----------
    strategy_name    : str
    G_template       : nx.Graph — topology + block_id attributes (never mutated)
    sim_structures   : 6-tuple from build_sim_structures(G_template), or None.
                       Pass this in to avoid rebuilding on every run.
    pretrained_gnts  : LocalGNTS | GlobalGNTS | None
    kits_schedule    : list[(start_day, kits)] or None
    """
    N = G_template.number_of_nodes()
    if N == 0:
        return [], None, {}, []

    # Build static structures if not supplied
    if sim_structures is None:
        sim_structures = build_sim_structures(G_template)

    node_index, nodes, adj_csr, block_arr, edge_index_t, block_id_t = sim_structures
    num_blocks = infer_num_blocks(block_arr)

    # Fresh state array — no deepcopy of G
    state_arr = initialize_epidemic(N, config.INITIAL_INFECTED)

    metrics = {
        "total_tests_administered": 0,
        "total_positive_tests":     0,
        "total_wasted_tests":       0,
        "peak_infections":          0,
        "time_to_peak":            -1,
        "integrated_infections":    0,
    }

    daily_records = []
    epoch_losses  = []
    test_buffer   = deque()

    # ----------------------
    # Strategy initialisation
    # ----------------------
    agent = None

    if strategy_name.startswith('LocalGNTS'):
        agent = copy.deepcopy(pretrained_gnts) if pretrained_gnts else LocalGNTS(
            G_template, num_blocks, config.GNN_OUTPUT_DIM,
            config.LOCAL_AGENT_CONTEXT_DIM, config.WEIGHT_DECAY,
        )

    elif strategy_name.startswith('GlobalGNTS'):
        if pretrained_gnts:
            agent = copy.deepcopy(pretrained_gnts)
        else:
            # Pass precomputed structures so GlobalGNTS never rebuilds them
            agent = GlobalGNTS(
                G_template, num_blocks, config.GNN_OUTPUT_DIM,
                config.GLOBAL_AGENT_CONTEXT_DIM, config.WEIGHT_DECAY,
                precomputed=sim_structures,
            )

    elif strategy_name.startswith('Beta'):
        mab = BetaBinomialMAB(num_blocks)

    if kits_schedule is None:
        kits_schedule = config.KITS_SCHEDULE

    # ----------------------
    # Simulation loop
    # ----------------------
    for day in range(config.SIMULATION_DAYS):

        # --------------------------------------------------
        # TESTING PHASE — before epidemic step
        # --------------------------------------------------
        allocations = np.zeros(num_blocks, dtype=int)

        if day >= config.TESTING_START_DAY and num_blocks > 0:
            kits_today = get_kits_for_day(day, kits_schedule)

            if kits_today > 0:

                if agent:
                    # Sync state array → G so GNN can read node states
                    sync_state_to_graph(G_template, state_arr, nodes)
                    props = agent.get_allocation_proportions(
                        G_template, test_buffer, day, config.SIMULATION_DAYS
                    )
                    props = (props / props.sum()
                             if props.sum() > 0
                             else np.ones(num_blocks) / num_blocks)
                    allocations = np.floor(props * kits_today).astype(int)
                    remainder   = kits_today - allocations.sum()
                    if remainder > 0:
                        residuals        = props * kits_today - allocations
                        allocations[np.argsort(residuals)[-remainder:]] += 1

                elif strategy_name.startswith('Beta'):
                    mab.update_priors(test_buffer)
                    for _ in range(kits_today):
                        arm = mab.select_arm()
                        if 0 <= arm < num_blocks:
                            allocations[arm] += 1

                else:
                    counts = []
                    for bid in range(num_blocks):
                        bstates = state_arr[block_arr == bid]
                        counts.append({
                            STATE_NAMES[st]: int((bstates == st).sum())
                            for st in range(6)
                        })
                    alloc_map = {
                        'Uniform':      uniform_allocation,
                        'Random':       random_allocation,
                        'Proportional': proportional_allocation,
                    }
                    allocations = alloc_map.get(
                        strategy_name, uniform_allocation
                    )(num_blocks, kits_today, current_counts=counts)

            # Testing — state_arr modified in-place
            daily_results = [{'positive': 0, 'negative': 0} for _ in range(num_blocks)]
            for bid in range(num_blocks):
                pos, wasted = node_level_testing(
                    state_arr, block_arr, bid, nodes, G_template, allocations[bid]
                )
                num_pos = sum(pos.values())
                daily_results[bid]['positive'] = num_pos
                daily_results[bid]['negative'] = wasted
                metrics['total_tests_administered'] += allocations[bid]
                metrics['total_positive_tests']     += num_pos
                metrics['total_wasted_tests']       += wasted

            # GNTS update — sync graph so GNN context is current
            if agent:
                sync_state_to_graph(G_template, state_arr, nodes)
                loss = agent.update(
                    G_template, day, config.SIMULATION_DAYS,
                    daily_results, history=test_buffer,
                )
                epoch_losses.append(loss)

            test_buffer.append(daily_results)

        # --------------------------------------------------
        # EPIDEMIC STEP — after testing
        # --------------------------------------------------
        run_seiqr_step(
            state_arr, adj_csr, block_arr,
            config.BETA, config.SIGMA, config.GAMMA,
            config.ASYMPTOMATIC_PROB,
            config.HUB_BLOCK_ID, config.HUB_BETA_MULTIPLIER,
            config.LONG_RANGE_INFECTION_PROB,
            config.WANING_IMMUNITY_PROB,
        )

        # --------------------------------------------------
        # Daily state aggregation — O(N) bincount
        # --------------------------------------------------
        c = np.bincount(state_arr.astype(np.int64), minlength=6)
        infectious_today = int(c[I] + c[A])

        daily_records.append({
            'Day': day,
            'S': int(c[S]), 'E': int(c[E]),
            'I': int(c[I]), 'A': int(c[A]),
            'Q': int(c[Q]), 'R': int(c[R]),
        })

        metrics['integrated_infections'] += infectious_today
        if infectious_today > metrics['peak_infections']:
            metrics['peak_infections'] = infectious_today
            metrics['time_to_peak']    = day

    return daily_records, agent, metrics, epoch_losses