# simulation_vax.py
# ---------------------------------------------------------------------------
# Vaccine-workflow simulation loop.
# Mirrors the structure of simulation.py with the following changes:
#
#   1. 'kits' -> 'doses' throughout
#   2. node_level_testing -> node_level_vaccination (random vaccination)
#   3. Immediate test reward -> lagged, global-trend-corrected reward
#      via VaxRewardBuffer
#   4. Epidemic step uses run_seiqr_vax_step (includes V state and
#      waning vaccine immunity)
#   5. State counting includes V compartment
#   6. Metrics updated for vaccine context:
#        - total_doses_administered
#        - total_productive_doses   (S/E -> V)
#        - total_wasted_doses       (A/V/R dosed)
#        - peak_infections
#        - total_new_infections
#        - herd_immunity_day        (first day V+R >= herd_threshold * n_nodes)
#        - integrated_infections
#   7. Strategy dispatch updated:
#        - LocalGNTSVax / GlobalGNTSVax (from gnts_vax)
#        - BetaBinomialMABVax / GammaPoissonMABVax (from strategies_vax)
#        - Uniform / Random / Proportional / RiskWeighted heuristics
#
# Public API (for main_vax.py):
#   run_simulation_vax(strategy_name, sim_graph, pretrained_gnts=None)
#       -> history, agent, metrics
# ---------------------------------------------------------------------------

import copy
import numpy as np
from collections import deque, Counter

import config
from netepi_vax import (
    S, E, I, A, V, R, Q,
    STATE_NAMES_VAX,
    make_initial_states_vax,
    run_seiqr_vax_step,
    build_sim_graph,
)
from strategies_vax import (
    BetaBinomialMABVax, GammaPoissonMABVax,
    uniform_allocation,
    random_allocation,
    proportional_allocation_vax,
    risk_weighted_allocation,
)
from gnts_vax import LocalGNTSVax, GlobalGNTSVax
from vaccination import (
    node_level_vaccination,
    VaxRewardBuffer,
    count_I_by_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_doses_for_day(day, schedule):
    """
    Return the number of doses available on a given day.
    Uses config.DOSES_SCHEDULE format: list of (start_day, num_doses) tuples.
    """
    if day < schedule[0][0]:
        return 0
    doses = 0
    for start_day, num_doses in schedule:
        if day >= start_day:
            doses = num_doses
    return doses


def _count_states_by_block(states, sim_graph):
    """
    Return list[Counter] indexed by block_id using vaccine state integers.
    Includes V compartment.
    """
    num_blocks = sim_graph['num_blocks']
    day_counts = [Counter() for _ in range(num_blocks)]
    for b in range(num_blocks):
        bn = sim_graph['block_nodes'][b]
        bs = states[bn]
        for st_int in [S, E, I, A, V, R]:          # Q excluded (unused in vax)
            cnt = int(np.sum(bs == st_int))
            if cnt:
                day_counts[b][STATE_NAMES_VAX[st_int]] = cnt
    return day_counts


def _is_gnts_vax(strategy_name):
    return (strategy_name.startswith('LocalGNTSVax')
            or strategy_name.startswith('GlobalGNTSVax'))


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation_vax(strategy_name, sim_graph, pretrained_gnts=None):
    """
    Run one vaccine-workflow simulation episode.

    Parameters
    ----------
    strategy_name   : str
        One of: 'LocalGNTSVax', 'GlobalGNTSVax',
                'BetaBinomialVax-<lookback>', 'GammaPoissonVax-<lookback>',
                'Uniform', 'Random', 'Proportional', 'RiskWeighted'
    sim_graph       : dict  — produced by build_sim_graph(), never mutated
    pretrained_gnts : LocalGNTSVax | GlobalGNTSVax | None

    Returns
    -------
    history : list[list[Counter]]   — [day][block] -> state counts
    agent   : LocalGNTSVax | GlobalGNTSVax | None
    metrics : dict
    """
    states     = make_initial_states_vax(sim_graph, config.VAX_INITIAL_INFECTED)
    num_blocks = sim_graph['num_blocks']
    n_nodes    = sim_graph['n_nodes']

    # Herd immunity threshold: fraction of population that must be
    # immune (V + R) for transmission to decline sustainably.
    # Default 0.7 — can be moved to config if needed.
    herd_threshold = getattr(config, 'HERD_IMMUNITY_THRESHOLD', 0.7)

    metrics = {
        'total_doses_administered': 0,
        'total_productive_doses':   0,     # S/E -> V
        'total_wasted_doses':       0,     # dosed A/V/R
        'peak_infections':          0,
        'time_to_peak':            -1,
        'total_new_infections':     0,
        'herd_immunity_day':       -1,
        'integrated_infections':    0,
        'daily_allocations':        [],
    }

    history = []

    # ---- Lookback window for MAB strategies --------------------------------
    lookback_days = 0
    if '-' in strategy_name:
        try:
            lookback_days = int(strategy_name.split('-')[-1])
        except ValueError:
            pass

    # historical_vax_results: deque of matured daily reward dicts
    # Used by MAB and GNTS agents (analogous to daily_test_results_buffer
    # in simulation.py, but contains only matured lagged rewards).
    historical_vax_results = deque(
        maxlen=lookback_days if lookback_days > 0 else None
    )

    # Lagged reward buffer — releases reward after VACCINE_EFFECT_LAG days
    reward_buffer = VaxRewardBuffer(num_blocks, config.VACCINE_EFFECT_LAG)

    # ---- Agent construction ------------------------------------------------
    agent = None
    gnts_kwargs = dict(
        sim_graph    = sim_graph,
        num_blocks   = num_blocks,
        gnn_out_dim  = config.GNN_OUTPUT_DIM,
        context_dim  = config.GNTS_CONTEXT_DIM,
        weight_decay = config.WEIGHT_DECAY,
    )

    if strategy_name.startswith('LocalGNTSVax'):
        if pretrained_gnts is not None:
            agent = copy.deepcopy(pretrained_gnts)
            agent.sim_graph = sim_graph
        else:
            agent = LocalGNTSVax(**gnts_kwargs)

    elif strategy_name.startswith('GlobalGNTSVax'):
        if pretrained_gnts is not None:
            agent = copy.deepcopy(pretrained_gnts)
            agent.sim_graph = sim_graph
        else:
            agent = GlobalGNTSVax(**gnts_kwargs)

    # ---- Simulation loop ---------------------------------------------------
    for day in range(config.SIMULATION_DAYS):

        # Count visible I per block — needed for reward buffer and heuristics
        I_counts = count_I_by_block(states, sim_graph)

        # ---- Vaccination phase ---------------------------------------------
        if day >= config.VACCINATION_START_DAY:
            doses_today = get_doses_for_day(day, config.DOSES_SCHEDULE)

            if doses_today > 0:
                # Record pre-vaccination I counts for lagged reward
                reward_buffer.record_vaccination_day(day, I_counts)

                # -- Across-block dose allocation ----------------------------
                current_counts = _count_states_by_block(states, sim_graph)

                if _is_gnts_vax(strategy_name):
                    proportions = agent.get_allocation_proportions(
                        states, sim_graph, historical_vax_results,
                        day, config.SIMULATION_DAYS
                    )
                    dose_allocations = np.floor(
                        proportions * doses_today
                    ).astype(int)
                    rem = doses_today - dose_allocations.sum()
                    if rem > 0:
                        residuals = (proportions * doses_today) - dose_allocations
                        for idx in np.argsort(residuals)[-rem:]:
                            dose_allocations[idx] += 1

                elif strategy_name.startswith('BetaBinomialVax'):
                    mab = BetaBinomialMABVax(num_blocks)
                    mab.update_priors(historical_vax_results)
                    dose_allocations = np.zeros(num_blocks, dtype=int)
                    for _ in range(doses_today):
                        dose_allocations[mab.select_arm()] += 1

                elif strategy_name.startswith('GammaPoissonVax'):
                    mab = GammaPoissonMABVax(num_blocks)
                    mab.update_priors(historical_vax_results, lookback_days)
                    dose_allocations = np.zeros(num_blocks, dtype=int)
                    for _ in range(doses_today):
                        dose_allocations[mab.select_arm()] += 1

                else:
                    dose_allocations = {
                        'Uniform':       uniform_allocation,
                        'Random':        random_allocation,
                        'Proportional':  proportional_allocation_vax,
                        'RiskWeighted':  risk_weighted_allocation,
                    }[strategy_name](
                        num_blocks, doses_today,
                        current_counts=current_counts
                    )

                # -- Within-block random vaccination -------------------------
                metrics['daily_allocations'].append(dose_allocations)

                for block_id in range(num_blocks):
                    productive, wasted = node_level_vaccination(
                        states, sim_graph, block_id,
                        dose_allocations[block_id]
                    )
                    metrics['total_doses_administered'] += dose_allocations[block_id]
                    metrics['total_productive_doses']   += productive
                    metrics['total_wasted_doses']       += wasted

        # ---- Check for matured lagged reward -------------------------------
        # Done after vaccination but before epidemic step so that the
        # post-vaccination state is not yet advanced. The reward compares
        # I counts at vaccination day (stored in buffer) vs I counts NOW
        # (lag days later, pre-epidemic-step of current day).
        matured_reward = reward_buffer.get_reward(day, I_counts)
        if matured_reward is not None:
            if _is_gnts_vax(strategy_name) and agent is not None:
                agent.update(
                    states, sim_graph, day, config.SIMULATION_DAYS,
                    matured_reward
                )
            historical_vax_results.append(matured_reward)

        # ---- Epidemic step -------------------------------------------------
        run_seiqr_vax_step(
            states, sim_graph,
            config.VAX_BETA,
            config.VAX_SIGMA,
            config.GAMMA,
            config.ASYMPTOMATIC_PROB,
            config.HUB_BLOCK_ID,
            config.HUB_BETA_MULTIPLIER,
            config.VAX_LONG_RANGE_INFECTION_PROB,
            config.WANING_IMMUNITY_PROB,
            config.WANING_VACCINE_PROB,
        )

        # ---- Daily metrics -------------------------------------------------
        day_counts       = _count_states_by_block(states, sim_graph)
        infectious_today = int(np.sum((states == I) | (states == A)))
        immune_today     = int(np.sum((states == V) | (states == R)))

        metrics['integrated_infections'] += infectious_today

        if infectious_today > metrics['peak_infections']:
            metrics['peak_infections'] = infectious_today
            metrics['time_to_peak']    = day

        if (metrics['herd_immunity_day'] == -1
                and immune_today >= herd_threshold * n_nodes):
            metrics['herd_immunity_day'] = day

        history.append(day_counts)

    return history, agent, metrics
