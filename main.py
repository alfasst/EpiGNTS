# main.py
# Description: Runs the full train-and-test cycle for SBM networks.
# This script now only outputs raw daily data and run-specific metrics.

import os
import torch
from tqdm import tqdm
import collections
import numpy as np
import pandas as pd
import csv

import config
from simulation import run_simulation
from network_epidemic import create_sbm_network
from strategies import LocalGNTS, average_gnts_bandits

# --- Configuration for output structure and network naming ---
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

NETWORK_SHORT_NAMES = {
    "Fragmented_1k": "SBM-1k",
    "Core-Periphery_2k": "SBM-2k",
    "Bimodal_3k": "SBM-3k",
    "Pyramidal_4k": "SBM-4k",
    "Heterogeneous_5k": "SBM-5k-Medium",
    "SBM_5k_Zero_Modularity": "SBM-5k-Zero",
    "SBM_5k_Low_Modularity": "SBM-5k-Low",
    "SBM_5k_Medium_Modularity": "SBM-5k-Medium",
    "SBM_5k_High_Modularity": "SBM-5k-High",
    "SBM_5k_Max_Modularity": "SBM-5k-Max",
}

def get_daily_efficiency(history):
    """Calculates quarantine efficiency for each day of a single simulation run."""
    daily_efficiency = []
    for day_data in history:
        total_counts = collections.Counter()
        for block_data in day_data:
            total_counts.update(block_data)
        i, e, a, q = [total_counts.get(s, 0) for s in ['I', 'E', 'A', 'Q']]
        denominator = i + e + a + q
        efficiency = q / denominator if denominator > 0 else 0
        daily_efficiency.append(efficiency)
    return daily_efficiency

def export_raw_daily_data(results_dict, network_name):
    """Saves raw daily efficiency from all runs to individual CSV files."""
    for strategy_name, histories_list in results_dict.items():
        all_runs_data = []
        for run_idx, history in enumerate(histories_list):
            efficiencies = get_daily_efficiency(history)
            for day, efficiency in enumerate(efficiencies):
                all_runs_data.append({'Day': day, 'Run': run_idx, 'Efficiency': efficiency})
        
        df = pd.DataFrame(all_runs_data)
        short_name = NETWORK_SHORT_NAMES.get(network_name, network_name)
        filename = os.path.join(RESULTS_DIR, f"{short_name}_{strategy_name}.csv")
        df.to_csv(filename, index=False)
    print(f" > Raw daily data exported for network '{network_name}'.")

def export_run_metrics(all_metrics, network_name):
    """NEW: Exports the detailed metrics from each individual test run to a CSV."""
    for strategy, metrics_list in all_metrics.items():
        df = pd.DataFrame(metrics_list)
        # We don't need to save the large daily_allocations array here
        df = df.drop(columns=['daily_allocations']) 
        
        short_name = NETWORK_SHORT_NAMES.get(network_name, network_name)
        filename = os.path.join(RESULTS_DIR, f"{short_name}_{strategy}_metrics.csv")
        df.to_csv(filename, index=False)
    print(f" > Run-specific metrics exported for network '{network_name}'.")


if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    summary_file_path = os.path.join(RESULTS_DIR, "strategy_summary.csv")
    # NEW: Check for completion by looking for the final raw data file of a run
    completed_networks = set()
    if os.path.exists(summary_file_path):
        try:
            summary_df = pd.read_csv(summary_file_path)
            completed_networks = set(summary_df['Network_Name'].unique())
        except (pd.errors.EmptyDataError, FileNotFoundError):
             # If file is empty or missing, start fresh
            completed_networks = set()


    for network_params in config.TEST_NETWORKS:
        network_name = network_params["name"]
        if network_name in completed_networks:
            print(f"\n--- Skipping already completed SBM network: {network_name} ---")
            continue
        
        print(f"\n--- Starting Experiment for SBM Network: {network_name} ---")
        
        G_template = create_sbm_network(
            network_params["block_sizes"], network_params["p_in"], network_params["p_out"]
        )
        config.NUM_BLOCKS = len(network_params["block_sizes"])
        config.BLOCK_SIZES = network_params["block_sizes"]

        short_name = NETWORK_SHORT_NAMES.get(network_name, network_name)
        model_path = os.path.join(RESULTS_DIR, f"agent_{short_name}.pth")
        master_agent = LocalGNTS(G_template, config.NUM_BLOCKS, config.LOCAL_GNN_OUTPUT_DIM,
                                  config.LOCAL_GNTS_CONTEXT_DIM, config.WEIGHT_DECAY)
        
        if os.path.exists(model_path):
            print(f"Found existing model. Loading weights from {model_path}")
            master_agent.load_model(model_path)
        else:
            print(f"No existing model found. Starting training for {network_name}...")
            trained_agents = []
            for i in tqdm(range(config.N_TRAINING_RUNS), desc=f"Training on {network_name}"):
                _, trained_agent, _ = run_simulation('LocalGNTS-14', G_template)
                trained_agents.append(trained_agent)
            
            master_agent = average_gnts_bandits(trained_agents, master_agent)
            master_agent.save_model(model_path)
            print(f"Training complete. Model saved to {model_path}")

        print(f"Starting testing for {network_name}...")
        strategies_to_run = ['LocalGNTS-14', 'Beta-Binomial-14', 'Gamma-Poisson-14',
                             'Proportional', 'Uniform', 'Random']
        
        all_test_results = collections.defaultdict(list)
        all_test_metrics = collections.defaultdict(list)
        
        for i in tqdm(range(config.N_TESTING_RUNS), desc=f"Testing on {network_name}"):
            for strategy in strategies_to_run:
                history, _, metrics = run_simulation(strategy, G_template, pretrained_gnts=master_agent)
                all_test_results[strategy].append(history)
                all_test_metrics[strategy].append(metrics)
        
        # --- MODIFIED: Exporting raw data and run metrics only ---
        export_raw_daily_data(all_test_results, network_name)
        export_run_metrics(all_test_metrics, network_name)

    print("\n✅ All SBM experiments are complete.")

