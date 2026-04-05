# simulation.py (UPDATED - FINAL)
# Aligned with main.py API (external agent + training_mode)
# Key features:
# - TEST → SPREAD → UPDATE
# - External agent injection (no internal GNTS creation)
# - training_mode controls weight updates
# - Sliding window buffer (14)
# - Vectorization preserved

import copy
from collections import deque
import numpy as np

import config
from network_epidemic import (
    S, E, I, A, Q, R, STATE_NAMES,
    build_sim_structures,
    initialize_epidemic,
    sync_state_to_graph,
    run_seiqr_step,
)
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
# Node-level testing
# --------------------------------------------------

def node_level_testing(state_arr, block_arr, block_id, nodes, G, num_kits):
    if num_kits <= 0:
        return {}, 0

    in_block = np.where(block_arr == block_id)[0]
    if in_block.size == 0:
        return {}, 0

    node_to_idx = {nodes[idx]: idx for idx in in_block}

    tier1 = [idx for idx in in_block if state_arr[idx] == I]
    tier1_set = set(tier1)

    tier2_set = set()
    for idx in tier1:
        for nbr in G.neighbors(nodes[idx]):
            nbr_idx = node_to_idx.get(nbr)
            if nbr_idx is not None and state_arr[nbr_idx] in (S, E, A):
                tier2_set.add(nbr_idx)

    tier3 = [
        idx for idx in in_block
        if state_arr[idx] in (S, E, A)
        and idx not in tier1_set
        and idx not in tier2_set
    ]

    queue = tier1 + list(tier2_set) + tier3

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
            state_arr[idx] = Q
        else:
            wasted += 1

        tested += 1

    return positives, wasted


# --------------------------------------------------
# Main simulation
# --------------------------------------------------

def run_simulation(strategy_name, G_template, sim_structures=None,
                   agent=None, kits_schedule=None, training_mode=False):

    N = G_template.number_of_nodes()
    if N == 0:
        return [], None, {}, []

    if sim_structures is None:
        sim_structures = build_sim_structures(G_template)

    node_index, nodes, adj_csr, block_arr, edge_index_t, block_id_t = sim_structures
    num_blocks = infer_num_blocks(block_arr)

    state_arr = initialize_epidemic(N, config.INITIAL_INFECTED)

    metrics = {
        "total_tests_administered": 0,
        "total_positive_tests": 0,
        "total_wasted_tests": 0,
        "peak_infections": 0,
        "time_to_peak": -1,
        "integrated_infections": 0,
    }

    daily_records = []
    epoch_losses = []

    # Sliding window (used by MAB only)
    test_buffer = deque(maxlen=14)

    # Prevent mutation during testing
    if agent is not None and not training_mode:
        agent = copy.deepcopy(agent)

    # MAB init (only if needed)
    mab = None
    if strategy_name.startswith('Beta'):
        mab = BetaBinomialMAB(num_blocks)

    if kits_schedule is None:
        kits_schedule = config.KITS_SCHEDULE

    # --------------------------------------------------
    # Simulation loop
    # --------------------------------------------------
    for day in range(config.SIMULATION_DAYS):

        allocations = np.zeros(num_blocks, dtype=int)

        # ----------------------
        # TEST PHASE
        # ----------------------
        if day >= config.TESTING_START_DAY and num_blocks > 0:

            kits_today = get_kits_for_day(day, kits_schedule)

            if kits_today > 0:

                if agent is not None:
                    sync_state_to_graph(G_template, state_arr, nodes)

                    props = agent.get_allocation_proportions(
                        G_template, day, config.SIMULATION_DAYS
                    )

                    props = props / props.sum() if props.sum() > 0 else np.ones(num_blocks) / num_blocks

                    allocations = np.floor(props * kits_today).astype(int)

                    remainder = kits_today - allocations.sum()
                    if remainder > 0:
                        residuals = props * kits_today - allocations
                        allocations[np.argsort(residuals)[-remainder:]] += 1

                elif mab is not None:
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
                        'Uniform': uniform_allocation,
                        'Random': random_allocation,
                        'Proportional': proportional_allocation,
                    }

                    allocations = alloc_map.get(
                        strategy_name, uniform_allocation
                    )(num_blocks, kits_today, current_counts=counts)

            daily_results = [{'positive': 0, 'negative': 0} for _ in range(num_blocks)]

            for bid in range(num_blocks):
                pos, wasted = node_level_testing(
                    state_arr, block_arr, bid, nodes, G_template, allocations[bid]
                )

                num_pos = sum(pos.values())
                daily_results[bid]['positive'] = num_pos
                daily_results[bid]['negative'] = wasted

                metrics['total_tests_administered'] += allocations[bid]
                metrics['total_positive_tests'] += num_pos
                metrics['total_wasted_tests'] += wasted

            test_buffer.append(daily_results)

        # ----------------------
        # SPREAD PHASE
        # ----------------------
        run_seiqr_step(
            state_arr, adj_csr, block_arr,
            config.BETA, config.SIGMA, config.GAMMA,
            config.ASYMPTOMATIC_PROB,
            config.HUB_BLOCK_ID, config.HUB_BETA_MULTIPLIER,
            config.LONG_RANGE_INFECTION_PROB,
            config.WANING_IMMUNITY_PROB,
        )

        # ----------------------
        # UPDATE PHASE
        # ----------------------
        if agent is not None and training_mode and day >= config.TESTING_START_DAY:
            sync_state_to_graph(G_template, state_arr, nodes)
            loss = agent.update(
                G_template, day, config.SIMULATION_DAYS,
                daily_results
            )
            epoch_losses.append(loss)

        # ----------------------
        # Metrics
        # ----------------------
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
            metrics['time_to_peak'] = day

    return daily_records, agent, metrics, epoch_losses
