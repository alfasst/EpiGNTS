# main.py
import csv
import collections
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

import config
from simulation import run_simulation

GPICKLE_DIR = "gpickle"

ALL_NETWORKS = config.SBM_NETWORKS + config.SNAP_NETWORKS

strategies_to_run = [
    "LocalGNTS-14",
    "Beta-Binomial-14",
    "Gamma-Poisson-14",
    "Proportional",
    "Uniform",
    "Random",
]


def export_daily_results(results, filename):
    header = [
        "Day",
        "Strategy",
        "S_avg",
        "E_avg",
        "I_avg",
        "A_avg",
        "Q_avg",
        "R_avg",
        "Efficiency_avg",
    ]
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for strategy_name, histories_list in results.items():
            avg_daily_counts = []
            for day in range(config.SIMULATION_DAYS):
                day_run_totals = []
                for history in histories_list:
                    run_total = collections.Counter()
                    for block_data in history[day]:
                        run_total.update(block_data)
                    day_run_totals.append(run_total)
                avg_counts_for_day = collections.Counter()
                for run_total in day_run_totals:
                    avg_counts_for_day.update(run_total)
                for k in avg_counts_for_day:
                    avg_counts_for_day[k] /= len(histories_list)
                avg_daily_counts.append(avg_counts_for_day)
            for day, counts in enumerate(avg_daily_counts):
                s, e, i, a, q, r = [
                    counts.get(st, 0) for st in ["S", "E", "I", "A", "Q", "R"]
                ]
                denominator = i + e + a + q
                efficiency = q / denominator if denominator > 0 else 0
                row = [day, strategy_name, s, e, i, a, q, r, efficiency]
                writer.writerow(row)


def plot_efficiency(results, filename):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))

    strat_name = {
        "LocalGNTS-14": "GNTS",
        "Beta-Binomial-14": "BBTS",
        "Gamma-Poisson-14": "GPTS",
        "Proportional": "Proportional",
        "Uniform": "Uniform",
        "Random": "Random",
    }

    for strategy_name, histories_list in results.items():
        all_runs_efficiencies = []
        for history in histories_list:
            quarantine_efficiency = []
            for day_data in history:
                total_counts = collections.Counter()
                for block_data in day_data:
                    total_counts.update(block_data)
                i, e, a, q = [total_counts.get(s, 0) for s in ["I", "E", "A", "Q"]]
                denominator = i + e + a + q
                efficiency = q / denominator if denominator > 0 else 0
                quarantine_efficiency.append(efficiency)
            all_runs_efficiencies.append(quarantine_efficiency)
        avg_efficiency = np.mean(all_runs_efficiencies, axis=0)
        label = strat_name.get(strategy_name, strategy_name)
        ax.plot(avg_efficiency, label=label, lw=2)

    ax.axvline(x=config.TESTING_START_DAY, color="black", linestyle="--")
    ax.set_xlabel("Day", fontsize=14)
    ax.set_ylabel("Avg. Quarantine Efficiency (Q / (I+E+A+Q))", fontsize=14)
    ax.legend(title="Strategy", fontsize=12, frameon=True)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


for net_config in ALL_NETWORKS:
    net_name = net_config["name"]
    gpickle_path = os.path.join(GPICKLE_DIR, f"{net_name}.gpickle")

    if not os.path.exists(gpickle_path):
        print(f"Skipping {net_name}: gpickle not found at {gpickle_path}")
        continue

    print(f"\nLoading network: {net_name}")
    G = nx.read_gpickle(gpickle_path)
    config.NUM_BLOCKS = len(set(nx.get_node_attributes(G, "block_id").values()))

    print(f"--- TRAINING PHASE FOR LocalGNTS on {net_name} ---")
    from strategies import LocalGNTS, average_gnts_bandits
    from simulation import run_simulation

    trained_agents = []
    for i in tqdm(range(config.N_TRAINING_RUNS), desc=f"Training {net_name}"):
        _, trained_agent, _ = run_simulation("LocalGNTS-14", G)
        trained_agents.append(trained_agent)

    master_agent_template = LocalGNTS(
        G,
        config.NUM_BLOCKS,
        config.LOCAL_GNN_OUTPUT_DIM,
        config.LOCAL_GNTS_CONTEXT_DIM,
        config.WEIGHT_DECAY,
    )
    master_agent = average_gnts_bandits(trained_agents, master_agent_template)

    print(f"\n--- TESTING PHASE FOR {net_name} ---")
    all_test_results = collections.defaultdict(list)
    all_test_metrics = collections.defaultdict(list)

    for i in tqdm(range(config.N_TESTING_RUNS), desc=f"Testing {net_name}"):
        for strategy in strategies_to_run:
            pretrained_model = (
                master_agent if strategy.startswith("LocalGNTS") else None
            )
            history, _, metrics = run_simulation(
                strategy, G, pretrained_gnts=pretrained_model
            )
            all_test_results[strategy].append(history)
            all_test_metrics[strategy].append(metrics)

    csv_filename = f"{net_name}_daily.csv"
    export_daily_results(all_test_results, csv_filename)
    print(f"Daily CSV exported: {csv_filename}")

    svg_filename = f"{net_name}.svg"
    plot_efficiency(all_test_results, svg_filename)
    print(f"Efficiency SVG exported: {svg_filename}")
