import os
import collections
import numpy as np
import networkx as nx
from tqdm import tqdm

# Import core simulation modules
import config
from network_epidemic import build_sim_graph
from simulation import run_simulation

# --- Experiment Setup ---
GPICKLE_DIR = "gpickle"
TARGET_NETWORKS = ["SBM-5k-Med", "SBM-5k-Zero", "SBM-5k-Low", "SBM-5k-High", "SBM-5k-Max"]
DRY_RUN_ITERATIONS = 10  # Run multiple times to smooth out stochastic noise

# We only need the biological parameters here. 
# We will completely bypass testing.
SCENARIOS = [
    {
        "name": "Peak_25",
        "BETA": 0.040,
        "INITIAL_INFECTED": 10,
        "LONG_RANGE_INFECTION_PROB": 0.0,
        "SIMULATION_DAYS": 150
    },
    {
        "name": "Peak_40",
        "BETA": 0.018,
        "INITIAL_INFECTED": 10,
        "LONG_RANGE_INFECTION_PROB": 0.0,
        "SIMULATION_DAYS": 150
    },
    {
        "name": "Peak_60",
        "BETA": 0.010,
        "INITIAL_INFECTED": 10,
        "LONG_RANGE_INFECTION_PROB": 0.0,
        "SIMULATION_DAYS": 200
    },
    {
        "name": "Peak_80",
        "BETA": 0.006,
        "INITIAL_INFECTED": 10,
        "LONG_RANGE_INFECTION_PROB": 0.0,
        "SIMULATION_DAYS": 200
    },
    {
        "name": "Peak_100",
        "BETA": 0.004,
        "INITIAL_INFECTED": 10,
        "LONG_RANGE_INFECTION_PROB": 0.0,
        "SIMULATION_DAYS": 250
    }
]

def get_daily_infections(history):
    """Extract total symptomatic (I) and asymptomatic (A) counts per day."""
    daily_I = []
    for day_data in history:
        # day_data is a list of Counters, one for each block
        total_infectious = sum(
            block_counts.get('I', 0) + block_counts.get('A', 0) 
            for block_counts in day_data
        )
        daily_I.append(total_infectious)
    return daily_I

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("EPIDEMIC DRY RUN: ESTABLISHING NATURAL PEAKS (NO INTERVENTION)")
    print(f"{'='*70}\n")
    
    # Bypass the testing block entirely
    config.TESTING_START_DAY = 9999
    config.KITS_SCHEDULE = []

    results = []

    for scenario in SCENARIOS:
        # Override config variables dynamically
        config.BETA = scenario["BETA"]
        config.INITIAL_INFECTED = scenario["INITIAL_INFECTED"]
        config.LONG_RANGE_INFECTION_PROB = scenario["LONG_RANGE_INFECTION_PROB"]
        config.SIMULATION_DAYS = scenario["SIMULATION_DAYS"]
        
        for net_name in TARGET_NETWORKS:
            gpickle_path = os.path.join(GPICKLE_DIR, f"{net_name}.gpickle")
            
            if not os.path.exists(gpickle_path):
                print(f"Skipping {net_name}: gpickle not found.")
                continue
                
            G_nx = nx.read_gpickle(gpickle_path)
            sim_graph = build_sim_graph(G_nx)
            config.NUM_BLOCKS = sim_graph['num_blocks']
            
            peak_days = []
            peak_heights = []
            
            # Run multiple stochastic iterations for a stable average
            for _ in range(DRY_RUN_ITERATIONS):
                # We pass "Uniform" just to satisfy the function signature, 
                # but because TESTING_START_DAY=9999, it is never actually executed.
                history, _, _ = run_simulation("Uniform", sim_graph, pretrained_gnts=None)
                
                daily_I = get_daily_infections(history)
                peak_day = np.argmax(daily_I)
                peak_height = daily_I[peak_day]
                
                peak_days.append(peak_day)
                peak_heights.append(peak_height)
                
            avg_peak_day = int(np.mean(peak_days))
            avg_peak_height = int(np.mean(peak_heights))
            
            results.append({
                "Scenario": scenario["name"],
                "Network": net_name,
                "Avg Peak Day": avg_peak_day,
                "Avg Peak Height": avg_peak_height
            })

    # --- Print Summary Table ---
    print(f"{'Scenario':<15} | {'Network':<15} | {'Avg Peak Day':<15} | {'Avg Peak Height':<15}")
    print("-" * 65)
    for r in results:
        print(f"{r['Scenario']:<15} | {r['Network']:<15} | {r['Avg Peak Day']:<15} | {r['Avg Peak Height']:<15}")
    print("-" * 65)