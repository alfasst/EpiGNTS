# simulation.py
# --------------------------------------------------
# Changes vs previous version:
# 1. Testing runs BEFORE epidemic step (quarantine before spread)
# 2. test_buffer always appended for all strategies
# 3. Lookback suffix parsing removed (unused)
# 4. agent.update() receives test_buffer for informed prior
# 5. GammaPoissonMAB removed
#
# Option 1 vectorisation changes:
# 6. G is no longer deep-copied each run — topology is shared read-only.
#    build_sim_structures() pre-computes the CSR adjacency and block array
#    once per graph; only the state array is allocated fresh per run.
# 7. State is maintained in a NumPy int8 array (state_arr) throughout.
#    G node 'state' attributes are only synced back via sync_state_to_graph()
#    immediately before the GNN context builder reads them.
# 8. node_level_testing() works directly on state_arr.
# 9. Daily state aggregation uses np.bincount on state_arr (O(N), no loop).
# --------------------------------------------------

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

def node_level_testing(state_arr, block_arr, block_id, node_index_inv, G, num_kits):
    """Test up to num_kits nodes in block_id.

    Priority order: confirmed I first, their S/E/A neighbours second,
    remaining S/E/A nodes third.

    state_arr is modified in-place (positives set to Q=4).

    Parameters
    ----------
    state_arr      : int8 array (N,) — live state
    block_arr      : int32 array (N,) — block id per node
    block_id       : int — which block to test
    node_index_inv : list — index -> original node id (for neighbour lookup)
    G              : nx.Graph — used only for neighbour lookup
    num_kits       : int

    Returns
    -------
    positives : dict {state_name: count}
    wasted    : int
    """
    if num_kits <= 0:
        return {}, 0

    in_block = np.where(block_arr == block_id)[0]
    if in_block.size == 0:
        return {}, 0

    block_set = set(in_block.tolist())

    # Tier 1: infectious I nodes in block
    tier1 = [idx for idx in in_block if state_arr[idx] == I]

    # Tier 2: S/E/A neighbours of tier1 that are also in block
    tier2 = []
    tier2_set = set()
    for idx in tier1:
        node = node_index_inv[idx]
        for nbr in G.neighbors(node):
            # node_index_inv is list indexed by contiguous idx;
            # we need the reverse: node -> idx, available via block_arr
            # Use G's node ordering — node ids may not equal indices.
            # We iterate by reconstructing idx from the neighbour node.
            pass  # handled below via node_to_idx

    # Build a local node->idx map for this block's neighbourhood
    node_to_idx = {node_index_inv[idx]: idx for idx in in_block}

    tier2_set = set()
    for idx in tier1:
        node = node_index_inv[idx]
        for nbr in G.neighbors(node):
            nbr_idx = node_to_idx.get(nbr)
            if nbr_idx is not None and state_arr[nbr_idx] in (S, E, A):
                tier2_set.add(nbr_idx)

    tier2 = list(tier2_set)

    # Tier 3: remaining S/E/A in block not in tier1 or tier2
    tier1_set = set(tier1)
    tier3 = [
        idx for idx in in_block
        if state_arr[idx] in (S, E, A) and idx not in tier1_set and idx not in tier2_set
    ]

    queue = tier1 + tier2 + tier3

    positives = {}
    wasted = 0
    tested = 0

    for idx in queue:
        if tested >= num_kits:
            break
        st = state_arr[idx]
        if st in (E, I, A):
            name = STATE_NAMES[st]
            positives[name] = positives.get(name, 0) + 1
            state_arr[idx] = Q        # quarantine in-place
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
    sim_structures   : tuple returned by build_sim_structures(G_template),
                       or None to build on the fly (slower — pass it in).
    pretrained_gnts  : LocalGNTS | GlobalGNTS | None
    kits_schedule    : list of (start_day, kits) tuples or None

    Returns
    -------
    daily_records, agent, metrics, epoch_losses
    """
    N = G_template.number_of_nodes()
    if N == 0:
        return [], None, {}, []

    # Pre-compute static structures if not supplied
    if sim_structures is None:
        sim_structures = build_sim_structures(G_template)
    node_index, nodes, adj_csr, block_arr = sim_structures

    num_blocks = infer_num_blocks(block_arr)

    # Fresh state array for this run — no deepcopy of G needed
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
        agent = (pretrained_gnts.__class__.__new__(pretrained_gnts.__class__)
                 if False else None)  # placeholder — deepcopy below
        import copy
        agent = copy.deepcopy(pretrained_gnts) if pretrained_gnts else LocalGNTS(
            G_template, num_blocks, config.GNN_OUTPUT_DIM,
            config.LOCAL_AGENT_CONTEXT_DIM, config.WEIGHT_DECAY,
        )
    elif strategy_name.startswith('GlobalGNTS'):
        import copy
        agent = copy.deepcopy(pretrained_gnts) if pretrained_gnts else GlobalGNTS(
            G_template, num_blocks, config.GNN_OUTPUT_DIM,
            config.GLOBAL_AGENT_CONTEXT_DIM, config.WEIGHT_DECAY,
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
        # TESTING PHASE — before epidemic step so quarantined
        # nodes do not spread on this day
        # --------------------------------------------------
        allocations = np.zeros(num_blocks, dtype=int)

        if day >= config.TESTING_START_DAY and num_blocks > 0:
            kits_today = get_kits_for_day(day, kits_schedule)

            if kits_today > 0:

                if agent:
                    # GNN needs G.nodes[n]['state'] — sync array → graph
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
                        residuals     = props * kits_today - allocations
                        idx           = np.argsort(residuals)[-remainder:]
                        allocations[idx] += 1

                elif strategy_name.startswith('Beta'):
                    mab.update_priors(test_buffer)
                    for _ in range(kits_today):
                        arm = mab.select_arm()
                        if 0 <= arm < num_blocks:
                            allocations[arm] += 1

                else:
                    # Heuristics read per-block state counts from state_arr
                    counts = []
                    for bid in range(num_blocks):
                        in_block = block_arr == bid
                        block_states = state_arr[in_block]
                        counts.append({
                            STATE_NAMES[st]: int((block_states == st).sum())
                            for st in range(6)
                        })

                    alloc_map = {
                        'Uniform':      uniform_allocation,
                        'Random':       random_allocation,
                        'Proportional': proportional_allocation,
                    }
                    func        = alloc_map.get(strategy_name, uniform_allocation)
                    allocations = func(num_blocks, kits_today, current_counts=counts)

            # Run testing — state_arr modified in-place
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

            # GNTS update — sync graph first so GNN context is current
            if agent:
                sync_state_to_graph(G_template, state_arr, nodes)
                loss = agent.update(
                    G_template, day, config.SIMULATION_DAYS,
                    daily_results, history=test_buffer,
                )
                epoch_losses.append(loss)

            test_buffer.append(daily_results)

        # --------------------------------------------------
        # EPIDEMIC STEP — after testing; state_arr modified in-place
        # --------------------------------------------------
        run_seiqr_step(
            state_arr,
            adj_csr,
            block_arr,
            config.BETA,
            config.SIGMA,
            config.GAMMA,
            config.ASYMPTOMATIC_PROB,
            config.HUB_BLOCK_ID,
            config.HUB_BETA_MULTIPLIER,
            config.LONG_RANGE_INFECTION_PROB,
            config.WANING_IMMUNITY_PROB,
        )

        # --------------------------------------------------
        # Daily state aggregation — O(N) bincount, no loop
        # --------------------------------------------------
        counts_today  = np.bincount(state_arr.astype(np.int64), minlength=6)
        infectious_today = int(counts_today[I] + counts_today[A])

        daily_records.append({
            'Day': day,
            'S':   int(counts_today[S]),
            'E':   int(counts_today[E]),
            'I':   int(counts_today[I]),
            'A':   int(counts_today[A]),
            'Q':   int(counts_today[Q]),
            'R':   int(counts_today[R]),
        })

        metrics['integrated_infections'] += infectious_today
        if infectious_today > metrics['peak_infections']:
            metrics['peak_infections'] = infectious_today
            metrics['time_to_peak']    = day

    return daily_records, agent, metrics, epoch_losses