# analyze.py
# Analysis for per-network–per-strategy daily CSVs
# --------------------------------------------------
# Design:
# - Each (Network, Strategy) has its own daily CSV
# - analyze.py DISCOVERS existing CSVs
# - Missing combinations are silently skipped
# - All summaries & plots are derived from daily data

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config


# Paths
RESULTS_DIR = os.path.join("results")
NET_METRICS_FILE = os.path.join("results", "network_metrics.csv")
PLOTS_DIR = os.path.join("plots")

os.makedirs(PLOTS_DIR, exist_ok=True)


# Canonical lists
NETWORKS = [n['name'] for n in config.TEST_NETWORKS] + \
           [n['name'] for n in config.SNAP_NETWORKS]

STRATEGIES = [
    "Uniform",
    "Random",
    "Proportional",
    "Beta-14",
    "LocalGNTS",
    "GlobalGNTS"
]


# Utilities
def load_existing_daily_csvs():
    """Load all existing (Network, Strategy) daily CSVs."""
    data = {}

    for net in NETWORKS:
        for strat in STRATEGIES:
            path = os.path.join(RESULTS_DIR, f"{net}__{strat}_daily.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                data[(net, strat)] = df
    return data


def add_derived_columns(df):
    df = df.copy()
    denom = df['E'] + df['I'] + df['A'] + df['Q']
    df['quarantine_efficiency'] = np.where(denom > 0, df['Q'] / denom, 0.0)
    df['infectious'] = df['I'] + df['A']
    df['positive_test_rate'] = (
        df['total_positive_tests'] /
        df['total_tests_administered'].replace(0, np.nan)
    ).fillna(0.0)
    return df


# Metric computations
def compute_summary(daily_data):
    """Compute Network–Strategy level summary from daily data."""
    records = []

    for (net, strat), df in daily_data.items():
        df = add_derived_columns(df)

        peak_val = df['infectious'].max()
        t_peak = df.loc[df['infectious'].idxmax(), 'Day']

        records.append({
            'Network': net,
            'Strategy': strat,
            'avg_quarantine_efficiency': df['quarantine_efficiency'].mean(),
            'peak_infections': peak_val,
            'time_to_peak': t_peak,
            'positive_test_rate': df['positive_test_rate'].iloc[0]
        })

    return pd.DataFrame(records)


# Plotting
def plot_qeff_vs_time(daily_data):
    """Plot quarantine efficiency vs time (one plot per network)."""

    for net in NETWORKS:
        fig, ax = plt.subplots()
        plotted = False

        for strat in STRATEGIES:
            key = (net, strat)
            if key not in daily_data:
                continue

            df = add_derived_columns(daily_data[key])
            ax.plot(df['Day'], df['quarantine_efficiency'], label=strat)
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_title(f"Quarantine Efficiency vs Time ({net})")
        ax.set_xlabel("Day")
        ax.set_ylabel("Q / (E+I+A+Q)")
        ax.legend()

        out = os.path.join(PLOTS_DIR, f"qeff_vs_time_{net}.svg")
        fig.savefig(out, format='svg')
        plt.close(fig)


def plot_qeff_vs_network_metrics(summary):
    net_metrics = pd.read_csv(NET_METRICS_FILE)
    merged = summary.merge(net_metrics, on='Network', how='left')

    for metric in ['Modularity', 'Clustering', 'Stiffness', 'Nodes']:
        fig, ax = plt.subplots()

        for strat in STRATEGIES:
            sub = merged[merged['Strategy'] == strat]
            if sub.empty:
                continue
            ax.scatter(sub[metric], sub['avg_quarantine_efficiency'], label=strat)

        ax.set_xlabel(metric)
        ax.set_ylabel("Average Quarantine Efficiency")
        ax.set_title(f"Quarantine Efficiency vs {metric}")
        ax.legend()

        out = os.path.join(PLOTS_DIR, f"qeff_vs_{metric.lower()}.svg")
        fig.savefig(out, format='svg')
        plt.close(fig)


# Main
if __name__ == '__main__':
    print("Loading existing daily CSVs...")
    daily_data = load_existing_daily_csvs()

    if not daily_data:
        raise RuntimeError("No daily CSVs found. Run main.py first.")

    print(f"Loaded {len(daily_data)} (network, strategy) datasets")

    print("Computing summary metrics...")
    summary = compute_summary(daily_data)

    summary_out = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    summary.to_csv(summary_out, index=False)
    print(f"Summary saved to {summary_out}")

    print("Generating plots...")
    plot_qeff_vs_time(daily_data)
    plot_qeff_vs_network_metrics(summary)

    print(f"All plots saved to {PLOTS_DIR}")
