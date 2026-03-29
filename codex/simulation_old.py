# simulation.py
# --------------------------------------------------
# Changes vs previous version:
# 1. Testing runs BEFORE epidemic step (quarantine before spread)
# 2. test_buffer always appended for all strategies
# 3. Lookback suffix parsing removed (unused)
# 4. agent.update() receives test_buffer for informed prior
# 5. GammaPoissonMAB removed
# --------------------------------------------------

import copy
from collections import deque, Counter
import numpy as np
import networkx as nx

import config
from network_epidemic import initialize_epidemic, run_seiqr_step
from gnts import LocalGNTS, GlobalGNTS
from strategies import (
    BetaBinomialMAB,
    uniform_allocation, random_allocation, proportional_allocation
)

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def infer_num_blocks(G):
    block_ids = nx.get_node_attributes(G, 'block_id')
    return max(block_ids.values()) + 1 if block_ids else 0


def get_kits_for_day(day, schedule):
    kits = 0
    for start_day, num in schedule:
        if day >= start_day:
            kits = num
    return kits


# --------------------------------------------------
# Node-level testing
# --------------------------------------------------

def node_level_testing(G, block_id, num_kits):
    if num_kits <= 0:
        return Counter(), 0

    block_nodes = [n for n, d in G.nodes(data=True) if d.get('block_id') == block_id]
    if not block_nodes:
        return Counter(), 0

    tier1 = {n for n in block_nodes if G.nodes[n].get('state') == 'I'}
    tier2 = set()
    for n in tier1:
        for nbr in G.neighbors(n):
            if nbr in block_nodes and G.nodes[nbr].get('state') in ('S', 'E', 'A'):
                tier2.add(nbr)

    tier3 = {n for n in block_nodes if G.nodes[n].get('state') in ('S', 'E', 'A')}
    tier3 -= tier1 | tier2

    queue = list(tier1) + list(tier2) + list(tier3)

    positives = Counter()
    wasted = 0
    tested = 0

    for node in queue:
        if tested >= num_kits:
            break
        state = G.nodes[node].get('state')
        if state in ('E', 'I', 'A'):
            positives[state] += 1
            G.nodes[node]['state'] = 'Q'
        else:
            wasted += 1
        tested += 1

    return positives, wasted


# --------------------------------------------------
# Main simulation routine
# --------------------------------------------------

def run_simulation(strategy_name, G_template, pretrained_gnts=None, kits_schedule=None):
    G = copy.deepcopy(G_template)
    if G.number_of_nodes() == 0:
        return [], None, {}, []

    initialize_epidemic(G, config.INITIAL_INFECTED)

    num_blocks = infer_num_blocks(G)

    metrics = {
        "total_tests_administered": 0,
        "total_positive_tests": 0,
        "total_wasted_tests": 0,
        "peak_infections": 0,
        "time_to_peak": -1,
        "integrated_infections": 0
    }

    daily_records = []
    history = []
    epoch_losses = []

    # ----------------------
    # Strategy initialization
    # ----------------------
    agent = None

    # Unbounded buffer — all testing history is retained
    test_buffer = deque()

    if strategy_name.startswith('LocalGNTS'):
        agent = copy.deepcopy(pretrained_gnts) if pretrained_gnts else LocalGNTS(
            G, num_blocks, config.GNN_OUTPUT_DIM,
            config.LOCAL_AGENT_CONTEXT_DIM, config.WEIGHT_DECAY
        )
    elif strategy_name.startswith('GlobalGNTS'):
        agent = copy.deepcopy(pretrained_gnts) if pretrained_gnts else GlobalGNTS(
            G, num_blocks, config.GNN_OUTPUT_DIM,
            config.GLOBAL_AGENT_CONTEXT_DIM, config.WEIGHT_DECAY
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
        # TESTING PHASE — runs first so quarantined nodes
        # do not spread on this day (fix #1)
        # --------------------------------------------------
        allocations = np.zeros(num_blocks, dtype=int)

        if day >= config.TESTING_START_DAY and num_blocks > 0:
            kits_today = get_kits_for_day(day, kits_schedule)

            if kits_today > 0:
                if agent:
                    props = agent.get_allocation_proportions(G, test_buffer, day, config.SIMULATION_DAYS)
                    props = props / props.sum() if props.sum() > 0 else np.ones(num_blocks) / num_blocks
                    allocations = np.floor(props * kits_today).astype(int)
                    remainder = kits_today - allocations.sum()
                    if remainder > 0:
                        residuals = props * kits_today - allocations
                        idx = np.argsort(residuals)[-remainder:]
                        allocations[idx] += 1

                elif strategy_name.startswith('Beta'):
                    mab.update_priors(test_buffer)
                    for _ in range(kits_today):
                        arm = mab.select_arm()
                        if 0 <= arm < num_blocks:
                            allocations[arm] += 1

                else:
                    counts = [Counter() for _ in range(num_blocks)]
                    for _, d in G.nodes(data=True):
                        bid = d.get('block_id')
                        if isinstance(bid, int) and 0 <= bid < num_blocks:
                            counts[bid][d.get('state')] += 1

                    alloc_map = {
                        'Uniform': uniform_allocation,
                        'Random': random_allocation,
                        'Proportional': proportional_allocation
                    }
                    func = alloc_map.get(strategy_name, uniform_allocation)
                    allocations = func(num_blocks, kits_today, current_counts=counts)

            daily_results = [{'positive': 0, 'negative': 0} for _ in range(num_blocks)]
            for bid in range(num_blocks):
                pos, wasted = node_level_testing(G, bid, allocations[bid])
                num_pos = sum(pos.values())
                daily_results[bid]['positive'] = num_pos
                daily_results[bid]['negative'] = wasted
                metrics['total_tests_administered'] += allocations[bid]
                metrics['total_positive_tests'] += num_pos
                metrics['total_wasted_tests'] += wasted

            # Pass test_buffer so update() trains against an informed prior (fix #4)
            if agent:
                loss = agent.update(G, day, config.SIMULATION_DAYS, daily_results, history=test_buffer)
                epoch_losses.append(loss)

            # Always append — keeps buffer populated for all strategies (fix #2)
            test_buffer.append(daily_results)

        # --------------------------------------------------
        # EPIDEMIC STEP — runs after testing so newly
        # quarantined nodes cannot spread today (fix #1)
        # --------------------------------------------------
        run_seiqr_step(
            G,
            config.BETA,
            config.SIGMA,
            config.GAMMA,
            config.ASYMPTOMATIC_PROB,
            config.HUB_BLOCK_ID,
            config.HUB_BETA_MULTIPLIER,
            config.LONG_RANGE_INFECTION_PROB,
            config.WANING_IMMUNITY_PROB
        )

        # ----------------------
        # Daily state aggregation
        # ----------------------
        state_counter = Counter()
        infectious_today = 0
        for _, d in G.nodes(data=True):
            s = d.get('state')
            state_counter[s] += 1
            if s in ('I', 'A'):
                infectious_today += 1

        daily_records.append({
            'Day': day,
            **{k: state_counter.get(k, 0) for k in ['S', 'E', 'I', 'A', 'Q', 'R']}
        })

        metrics['integrated_infections'] += infectious_today
        if infectious_today > metrics['peak_infections']:
            metrics['peak_infections'] = infectious_today
            metrics['time_to_peak'] = day

        history.append(state_counter)

    return daily_records, agent, metrics, epoch_losses