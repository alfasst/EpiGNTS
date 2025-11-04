# simulation.py

import copy
from collections import deque, Counter
import numpy as np

import config
from network_epidemic import initialize_epidemic, run_seiqr_step
from strategies import (
    LocalGNTS, BetaBinomialMAB, GammaPoissonMAB,
    uniform_allocation, random_allocation, proportional_allocation
)

def get_kits_for_day(day, schedule):
    """Determines the number of test kits available on a given day."""
    if day < schedule[0][0]: return 0
    kits = 0
    for start_day, num_kits in schedule:
        if day >= start_day:
            kits = num_kits
    return kits

def node_level_testing(G, block_id, num_kits):
    """Performs targeted testing within a specific block."""
    if num_kits == 0:
        return Counter(), 0

    block_nodes = [n for n, d in G.nodes(data=True) if d['block_id'] == block_id]
    
    tier1 = {n for n in block_nodes if G.nodes[n]['state'] == 'I'}
    tier2 = {neighbor for infected in tier1 for neighbor in G.neighbors(infected) 
             if G.nodes[neighbor]['block_id'] == block_id and G.nodes[neighbor]['state'] in ['S', 'E', 'A']}
    
    tested_or_higher_priority = tier1.union(tier2)
    tier3 = {n for n in block_nodes if G.nodes[n]['state'] in ['S', 'E', 'A']} - tested_or_higher_priority
    
    nodes_to_test = list(tier1) + list(tier2) + list(tier3)
    
    positive_counts = Counter()
    wasted_tests = 0
    
    for node in nodes_to_test[:num_kits]:
        state = G.nodes[node]['state']
        if state in ['E', 'I', 'A']:
            positive_counts[state] += 1
            G.nodes[node]['state'] = 'Q'
        elif state in ['S', 'R']:
            wasted_tests += 1
            
    return positive_counts, wasted_tests

def run_simulation(strategy_name, G_template, pretrained_gnts=None):
    """Runs a complete epidemic simulation for a given strategy."""
    G = copy.deepcopy(G_template)
    initialize_epidemic(G, config.INITIAL_INFECTED)
    
    metrics = {
        "total_new_infections": 0, "total_tests_administered": 0, "total_positive_tests": 0, 
        "total_wasted_tests": 0, "peak_infections": 0, "time_to_peak": -1, 
        "daily_allocations": [], "first_infection_day": {b: -1 for b in range(config.NUM_BLOCKS)}, 
        "first_intervention_day": {b: -1 for b in range(config.NUM_BLOCKS)}, "integrated_infections": 0
    }
    
    history = []
    
    lookback_days = 0
    if '-' in strategy_name:
        try: lookback_days = int(strategy_name.split('-')[-1])
        except (ValueError, IndexError): pass

    daily_test_results_buffer = deque(maxlen=lookback_days if lookback_days > 0 else None)
    
    agent = None
    if strategy_name.startswith('LocalGNTS'):
        if pretrained_gnts:
            agent = copy.deepcopy(pretrained_gnts)
            agent.G = G
        else:
            agent = LocalGNTS(G, config.NUM_BLOCKS, config.LOCAL_GNN_OUTPUT_DIM, 
                              config.LOCAL_GNTS_CONTEXT_DIM, config.WEIGHT_DECAY)

    for day in range(config.SIMULATION_DAYS):
        run_seiqr_step(G, config.BETA, config.SIGMA, config.GAMMA, config.ASYMPTOMATIC_PROB, 
                       config.HUB_BLOCK_ID, config.HUB_BETA_MULTIPLIER, 
                       config.LONG_RANGE_INFECTION_PROB, config.WANING_IMMUNITY_PROB)
        
        if day >= config.TESTING_START_DAY:
            kits_today = get_kits_for_day(day, config.KITS_SCHEDULE)
            
            if strategy_name.startswith('LocalGNTS'):
                proportions = agent.get_allocation_proportions(G, daily_test_results_buffer, day, config.SIMULATION_DAYS)
                kit_allocations = np.floor(proportions * kits_today).astype(int)
                rem_kits = kits_today - np.sum(kit_allocations)
                if rem_kits > 0:
                    residuals = (proportions * kits_today) - kit_allocations
                    for i in np.argsort(residuals)[-rem_kits:]: kit_allocations[i] += 1
            elif strategy_name.startswith('Beta') or strategy_name.startswith('Gamma'):
                mab = BetaBinomialMAB(config.NUM_BLOCKS) if strategy_name.startswith('Beta') else GammaPoissonMAB(config.NUM_BLOCKS)
                mab.update_priors(daily_test_results_buffer)
                kit_allocations = np.zeros(config.NUM_BLOCKS, dtype=int)
                for _ in range(kits_today):
                    chosen_block = mab.select_arm()
                    kit_allocations[chosen_block] += 1
            else:
                current_counts = [Counter() for _ in range(config.NUM_BLOCKS)]
                for _, data in G.nodes(data=True): current_counts[data['block_id']][data['state']] += 1
                alloc_func = {'Uniform': uniform_allocation, 'Random': random_allocation, 'Proportional': proportional_allocation}[strategy_name]
                kit_allocations = alloc_func(config.NUM_BLOCKS, kits_today, current_counts=current_counts)
            
            metrics["daily_allocations"].append(kit_allocations)
            daily_test_results = [{'positive': 0, 'negative': 0} for _ in range(config.NUM_BLOCKS)]
            
            for block_id in range(config.NUM_BLOCKS):
                pos_counts, wasted = node_level_testing(G, block_id, kit_allocations[block_id])
                num_pos = sum(pos_counts.values())
                daily_test_results[block_id]['positive'] = num_pos
                daily_test_results[block_id]['negative'] = wasted
                metrics["total_tests_administered"] += kit_allocations[block_id]
                metrics["total_positive_tests"] += num_pos
                metrics["total_wasted_tests"] += wasted
            
            if strategy_name.startswith('LocalGNTS'):
                agent.update(G, day, config.SIMULATION_DAYS, daily_test_results)
            
            daily_test_results_buffer.append(daily_test_results)

        day_counts = [Counter() for _ in range(config.NUM_BLOCKS)]
        total_infectious_today = 0
        for _, data in G.nodes(data=True): 
            day_counts[data['block_id']][data['state']] += 1
            if data['state'] in ['I', 'A']:
                total_infectious_today += 1
        
        metrics["integrated_infections"] += total_infectious_today
        if total_infectious_today > metrics["peak_infections"]:
            metrics["peak_infections"] = total_infectious_today
            metrics["time_to_peak"] = day
        history.append(day_counts)
        
    return history, agent, metrics

