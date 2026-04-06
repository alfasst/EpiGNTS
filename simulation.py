# simulation.py

import copy
import numpy as np
import random
from collections import deque, Counter

import config
from network_epidemic import (
    S, E, I, A, Q, R, STATE_NAMES,
    make_initial_states, run_seiqr_step, build_sim_graph
)
from strategies import (
    LocalGNTS, BetaBinomialMAB, GammaPoissonMAB,
    uniform_allocation, random_allocation, proportional_allocation
)


def get_kits_for_day(day, schedule):
    if day < schedule[0][0]:
        return 0
    kits = 0
    for start_day, num_kits in schedule:
        if day >= start_day:
            kits = num_kits
    return kits


def node_level_testing(states, sim_graph, block_id, num_kits):
    """
    Tiered testing within a block.
    Works directly on the integer state array and precomputed block/neighbour data.
    Returns (positive_counts: Counter[str->int], wasted_tests: int).
    """
    if num_kits == 0:
        return Counter(), 0

    block_nodes = sim_graph['block_nodes'][block_id]   # precomputed np.ndarray
    adj_lists   = sim_graph['adj_lists']
    block_set   = set(block_nodes.tolist())

    block_states = states[block_nodes]

    # Tier 1 — symptomatic infected in block
    tier1 = set(block_nodes[block_states == I].tolist())

    # Tier 2 — susceptible/exposed/asymptomatic neighbours of tier1 inside block
    tier2 = set()
    for node in tier1:
        for nb in adj_lists[node]:
            if nb in block_set and states[nb] in (S, E, A):
                tier2.add(int(nb))
    tier2 -= tier1

    # Tier 3 — remaining S/E/A in block
    t12 = tier1 | tier2
    tier3 = {int(n) for n in block_nodes
             if states[n] in (S, E, A) and n not in t12}

    nodes_to_test = list(tier1) + list(tier2) + list(tier3)

    positive_counts = Counter()
    wasted_tests = 0
    for node in nodes_to_test[:num_kits]:
        st = states[node]
        if st in (E, I, A):
            positive_counts[STATE_NAMES[st]] += 1
            states[node] = Q
        elif st in (S, R):
            wasted_tests += 1

    return positive_counts, wasted_tests


def _count_states_by_block(states, sim_graph):
    """Return list[Counter] indexed by block_id using integer states."""
    num_blocks  = sim_graph['num_blocks']
    block_ids   = sim_graph['block_ids']
    day_counts  = [Counter() for _ in range(num_blocks)]
    for b in range(num_blocks):
        bn = sim_graph['block_nodes'][b]
        bs = states[bn]
        for st_int in range(6):
            cnt = int(np.sum(bs == st_int))
            if cnt:
                day_counts[b][STATE_NAMES[st_int]] = cnt
    return day_counts


def run_simulation(strategy_name, sim_graph, pretrained_gnts=None):
    """
    Run one simulation episode.

    Parameters
    ----------
    strategy_name  : str
    sim_graph      : dict produced by build_sim_graph() — shared, never mutated
    pretrained_gnts: optional pre-trained LocalGNTS agent

    Returns
    -------
    history : list[list[Counter]]   — [day][block] -> state counts
    agent   : LocalGNTS | None
    metrics : dict
    """
    # --- Reset: only the state array, no deepcopy of the whole graph ---
    states = make_initial_states(sim_graph, config.INITIAL_INFECTED)

    num_blocks = sim_graph['num_blocks']

    metrics = {
        "total_new_infections":   0,
        "total_tests_administered": 0,
        "total_positive_tests":   0,
        "total_wasted_tests":     0,
        "peak_infections":        0,
        "time_to_peak":          -1,
        "daily_allocations":      [],
        "first_infection_day":    {b: -1 for b in range(num_blocks)},
        "first_intervention_day": {b: -1 for b in range(num_blocks)},
        "integrated_infections":  0,
    }

    history = []

    lookback_days = 0
    if '-' in strategy_name:
        try:
            lookback_days = int(strategy_name.split('-')[-1])
        except ValueError:
            pass

    daily_test_results_buffer = deque(maxlen=lookback_days if lookback_days > 0 else None)

    agent = None
    if strategy_name.startswith('LocalGNTS'):
        if pretrained_gnts:
            agent = copy.deepcopy(pretrained_gnts)
            agent.sim_graph = sim_graph
        else:
            agent = LocalGNTS(
                sim_graph, num_blocks,
                config.LOCAL_GNN_OUTPUT_DIM,
                config.LOCAL_GNTS_CONTEXT_DIM,
                config.WEIGHT_DECAY,
            )

    for day in range(config.SIMULATION_DAYS):

        if day >= config.TESTING_START_DAY:
            kits_today = get_kits_for_day(day, config.KITS_SCHEDULE)

            # ---- Kit allocation ----------------------------------------
            if strategy_name.startswith('LocalGNTS'):
                proportions = agent.get_allocation_proportions(
                    states, sim_graph, daily_test_results_buffer,
                    day, config.SIMULATION_DAYS
                )
                kit_allocations = np.floor(proportions * kits_today).astype(int)
                rem_kits = kits_today - np.sum(kit_allocations)
                if rem_kits > 0:
                    residuals = (proportions * kits_today) - kit_allocations
                    for idx in np.argsort(residuals)[-rem_kits:]:
                        kit_allocations[idx] += 1

            elif strategy_name.startswith('Beta') or strategy_name.startswith('Gamma'):
                if strategy_name.startswith('Beta'):
                    mab = BetaBinomialMAB(num_blocks)
                    mab.update_priors(daily_test_results_buffer)
                else:
                    mab = GammaPoissonMAB(num_blocks)
                    mab.update_priors(daily_test_results_buffer, lookback_days)
                kit_allocations = np.zeros(num_blocks, dtype=int)
                for _ in range(kits_today):
                    kit_allocations[mab.select_arm()] += 1

            else:  # Heuristics
                current_counts = _count_states_by_block(states, sim_graph)
                kit_allocations = {
                    'Uniform':      uniform_allocation,
                    'Random':       random_allocation,
                    'Proportional': proportional_allocation,
                }[strategy_name](num_blocks, kits_today, current_counts=current_counts)

            # ---- Testing -----------------------------------------------
            metrics["daily_allocations"].append(kit_allocations)
            daily_test_results = [{'positive': 0, 'negative': 0} for _ in range(num_blocks)]

            for block_id in range(num_blocks):
                pos_counts, wasted = node_level_testing(
                    states, sim_graph, block_id, kit_allocations[block_id]
                )
                num_pos = sum(pos_counts.values())
                daily_test_results[block_id]['positive'] = num_pos
                daily_test_results[block_id]['negative'] = wasted
                metrics["total_tests_administered"] += kit_allocations[block_id]
                metrics["total_positive_tests"]     += num_pos
                metrics["total_wasted_tests"]       += wasted

            if strategy_name.startswith('LocalGNTS'):
                agent.update(states, sim_graph, day, config.SIMULATION_DAYS, daily_test_results)

            daily_test_results_buffer.append(daily_test_results)

        # ---- Epidemic step -------------------------------------------------
        run_seiqr_step(
            states, sim_graph,
            config.BETA, config.SIGMA, config.GAMMA, config.ASYMPTOMATIC_PROB,
            config.HUB_BLOCK_ID, config.HUB_BETA_MULTIPLIER,
            config.LONG_RANGE_INFECTION_PROB, config.WANING_IMMUNITY_PROB,
        )

        # ---- Daily metrics -------------------------------------------------
        day_counts = _count_states_by_block(states, sim_graph)
        infectious_today = int(np.sum((states == I) | (states == A)))
        metrics["integrated_infections"] += infectious_today
        if infectious_today > metrics["peak_infections"]:
            metrics["peak_infections"] = infectious_today
            metrics["time_to_peak"]    = day
        history.append(day_counts)

    return history, agent, metrics