"""
expt_phase_diagram.py
---------------------
Generates a 2D Phase Diagram of Algorithmic Dominance.
Maps Epidemic Velocity (BETA) vs. Network Modularity (SBM-5k variants).
Exports a CSV of the results and a diverging SVG heatmap.
"""

import os
import csv
import collections

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt


# Import core simulation modules
import config
from network_epidemic import build_sim_graph
from simulation import run_simulation
from gnts import LocalGNTS, GlobalGNTS, average_gnts_bandits

# --- Experiment Setup ---
GPICKLE_DIR = "gpickle"
OUT_DIR = "csvs"
PLOT_DIR = 'plots'

# Y-Axis: Network Modularity (Ordered from lowest modularity to highest)
NETWORKS = [
    "SBM-5k-Zero",  # p_out = 0.01
    "SBM-5k-Low",   # p_out = 0.0075
    "SBM-5k-Med",   # p_out = 0.005
    "SBM-5k-High",  # p_out = 0.001
    "SBM-5k-Max"    # p_out = 0.0
]

# X-Axis: Epidemic Velocity (Beta parameter and corresponding Simulation Days)
VELOCITY_SCENARIOS = [
    {"BETA": 0.040, "SIM_DAYS": 150, "LABEL": "Fast (0.040)"},
    {"BETA": 0.018, "SIM_DAYS": 150, "LABEL": "Med-Fast (0.018)"},
    {"BETA": 0.010, "SIM_DAYS": 200, "LABEL": "Medium (0.010)"},
    {"BETA": 0.006, "SIM_DAYS": 200, "LABEL": "Med-Slow (0.006)"},
    {"BETA": 0.004, "SIM_DAYS": 250, "LABEL": "Slow (0.004)"}
]

STRATEGIES = [
    "LocalGNTS-14", "GlobalGNTS-14", 
    "Beta-Binomial-14", "Proportional", "Uniform", "Random"
]

GNTS_MODELS = ["LocalGNTS-14", "GlobalGNTS-14"]
HEURISTICS = ["Beta-Binomial-14", "Proportional", "Uniform", "Random"]

# --- Helpers ---
def save_gnts_model(agent, path):
    if isinstance(agent, LocalGNTS):
        payload = {'type': 'LocalGNTS', 'encoders': [m.state_dict() for m in agent.encoders], 'prior_boosters': [m.state_dict() for m in agent.prior_boosters]}
    elif isinstance(agent, GlobalGNTS):
        payload = {'type': 'GlobalGNTS', 'encoder': agent.encoder.state_dict(), 'prior_boosters': [m.state_dict() for m in agent.prior_boosters]}
    else:
        raise TypeError("save_gnts_model: unknown agent type")
    torch.save(payload, path)

def _train_gnts(strategy_name, sim_graph, agent_cls, agent_kwargs):
    trained_agents = []
    for _ in range(config.N_TRAINING_RUNS):
        _, agent, _ = run_simulation(strategy_name, sim_graph)
        trained_agents.append(agent)
    master_template = agent_cls(**agent_kwargs)
    return average_gnts_bandits(trained_agents, master_template)

def compute_efficiency(history):
    """Calculates the average detection ratio (Q / (I + E + A + Q)) across the run."""
    day_efficiencies = []
    for day_data in history:
        agg = collections.Counter()
        for block_data in day_data:
            agg.update(block_data)
        s, e, i, a, q, r = [agg.get(st, 0) for st in ["S", "E", "I", "A", "Q", "R"]]
        denominator = i + e + a + q
        eff = q / denominator if denominator > 0 else 0
        day_efficiencies.append(eff)
    return np.mean(day_efficiencies)


os.makedirs(OUT_DIR, exist_ok=True)

# Lock biological parameters for clean structural reads
config.INITIAL_INFECTED = 10
config.LONG_RANGE_INFECTION_PROB = 0.0

phase_data = []

print(f"\n{'='*70}")
print("STARTING 2D PHASE DIAGRAM EXPERIMENT")
print(f"{'='*70}\n")

for scenario in VELOCITY_SCENARIOS:
    config.BETA = scenario["BETA"]
    config.SIMULATION_DAYS = scenario["SIM_DAYS"]
    
    for net_name in NETWORKS:
        print(f"\nEvaluating: Velocity {scenario['LABEL']} | Network {net_name}")
        
        gpickle_path = os.path.join(GPICKLE_DIR, f"{net_name}.gpickle")
        if not os.path.exists(gpickle_path):
            print(f"Skipping {net_name}: gpickle not found.")
            continue

        G_nx = nx.read_gpickle(gpickle_path)
        sim_graph = build_sim_graph(G_nx)
        config.NUM_BLOCKS = sim_graph['num_blocks']

        gnts_kwargs = dict(
            sim_graph=sim_graph, num_blocks=config.NUM_BLOCKS,
            gnn_out_dim=config.GNN_OUTPUT_DIM, context_dim=config.GNTS_CONTEXT_DIM,
            weight_decay=config.WEIGHT_DECAY
        )

        # 1. Train GNTS Models
        local_master = _train_gnts("LocalGNTS-14", sim_graph, LocalGNTS, gnts_kwargs)
        global_master = _train_gnts("GlobalGNTS-14", sim_graph, GlobalGNTS, gnts_kwargs)

        # 2. Test All Strategies
        strategy_scores = {}
        for strategy in STRATEGIES:
            pretrained = None
            if strategy == "LocalGNTS-14": pretrained = local_master
            elif strategy == "GlobalGNTS-14": pretrained = global_master
            
            efficiencies = []
            for _ in range(config.N_TESTING_RUNS):
                history, _, _ = run_simulation(strategy, sim_graph, pretrained_gnts=pretrained)
                efficiencies.append(compute_efficiency(history))
            
            strategy_scores[strategy] = np.mean(efficiencies)

        # 3. Compute Phase Dominance
        best_gnts = max(strategy_scores[s] for s in GNTS_MODELS)
        best_heuristic = max(strategy_scores[s] for s in HEURISTICS)
        gnts_advantage = best_gnts - best_heuristic

        phase_data.append({
            "Velocity (Beta)": scenario["LABEL"],
            "Modularity (Network)": net_name,
            "Best GNTS Ratio": best_gnts,
            "Best Heuristic Ratio": best_heuristic,
            "GNTS Advantage": gnts_advantage
        })

# --- Export & Plot ---
df = pd.DataFrame(phase_data)
csv_path = os.path.join(OUT_DIR, "phase_diagram_data.csv")
df.to_csv(csv_path, index=False)
print(f"\nData exported to {csv_path}")

# Reload DF
df = pd.read_csv(csv_path)

# Pivot for Heatmap
pivot_df = df.pivot(index="Modularity (Network)", columns="Velocity (Beta)", values="GNTS Advantage")

# Reorder axes logically
pivot_df = pivot_df.reindex(NETWORKS[::-1]) # Max modularity at the top
pivot_df = pivot_df[[s["LABEL"] for s in VELOCITY_SCENARIOS]] # Fast to Slow

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Diverging colormap: Red = GNTS Wins, Blue = Heuristics Win
max_abs_val = max(abs(pivot_df.min().min()), abs(pivot_df.max().max()), 0.01)

sns.heatmap(
    pivot_df, 
    annot=True, 
    fmt=".3f", 
    cmap="RdBu_r", 
    center=0,
    vmin=-max_abs_val, 
    vmax=max_abs_val, 
    ax=ax,
    cbar_kws={'label': 'GNTS Advantage (Δ Detection Ratio)'}
)

ax.set_title("Phase Diagram: Algorithmic Dominance\n(Epidemic Velocity vs. Network Modularity)")
ax.set_xlabel("Epidemic Velocity (Fast $\\rightarrow$ Slow)")
ax.set_ylabel("Network Modularity (High $\\rightarrow$ Low)")

plt.xticks(rotation=30, ha='right')

svg_path = os.path.join(OUT_DIR, "phase_diagram.svg")
fig.savefig(svg_path, format="svg", bbox_inches="tight")
plt.close(fig)
print(f"Phase diagram SVG rendered to {svg_path}")