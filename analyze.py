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
RESULTS_DIR      = "csvs"
NET_METRICS_FILE = os.path.join("csvs", "network_metrics.csv")
PLOTS_DIR        = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)


# Canonical lists
NETWORKS = [n['name'] for n in config.SBM_NETWORKS] + \
           [n['name'] for n in config.SNAP_NETWORKS]

STRATEGIES = [
    "LocalGNTS-14", "GlobalGNTS-14",
    "Beta-Binomial-14",
    "Proportional",
    "Uniform",
    "Random",
]


def _strategy_to_filename(strat):
    """Match the naming convention used in main.py."""
    return strat.replace("-", "_")


# Utilities
def load_existing_daily_csvs():
    """Load all existing (Network, Strategy) daily CSVs."""
    data = {}
    for net in NETWORKS:
        for strat in STRATEGIES:
            path = os.path.join(RESULTS_DIR,
                                f"{net}_{_strategy_to_filename(strat)}.csv")
            if os.path.exists(path):
                data[(net, strat)] = pd.read_csv(path)
    return data


def add_derived_columns(df):
    """
    Add computed columns used by both summary and plotting.
    CSV columns available: Day, S_avg, E_avg, I_avg, A_avg, Q_avg, R_avg,
                           Efficiency_avg
    """
    df = df.copy()
    denom = df['E_avg'] + df['I_avg'] + df['A_avg'] + df['Q_avg']
    df['quarantine_efficiency'] = np.where(denom > 0, df['Q_avg'] / denom, 0.0)
    df['infectious']            = df['I_avg'] + df['A_avg']
    # Positive test rate: use the pre-computed Efficiency_avg column
    # (Q / (E+I+A+Q) already stored) as the proxy for positive test rate,
    # since raw test counts are not stored in the daily CSVs.
    df['positive_test_rate']    = df['Efficiency_avg']
    return df


# Metric computations
def compute_summary(daily_data):
    """Compute Network–Strategy level summary from daily data."""
    records = []

    for (net, strat), df in daily_data.items():
        df = add_derived_columns(df)

        peak_idx = df['infectious'].idxmax()
        peak_val = df['infectious'].max()
        t_peak   = df.loc[peak_idx, 'Day']

        records.append({
            'Network':                  net,
            'Strategy':                 strat,
            'avg_quarantine_efficiency': round(df['quarantine_efficiency'].mean(), 4),
            'peak_infections':           round(peak_val, 2),
            'time_to_peak':              int(t_peak),
            'positive_test_rate':        round(df['positive_test_rate'].mean(), 4),
        })

    return pd.DataFrame(records)


# Plotting
def plot_qeff_vs_time(daily_data):
    """One plot per network: quarantine efficiency over time for all strategies."""
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
        print(f"  Saved {out}")


def plot_qeff_vs_network_metrics(summary):
    """Scatter plots of avg quarantine efficiency against network-level metrics."""
    if not os.path.exists(NET_METRICS_FILE):
        print(f"  Skipping network-metric plots: {NET_METRICS_FILE} not found.")
        return

    net_metrics = pd.read_csv(NET_METRICS_FILE)
    merged      = summary.merge(net_metrics, on='Network', how='left')

    for metric in ['Modularity', 'Clustering', 'Stiffness', 'Nodes']:
        if metric not in merged.columns:
            continue
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
        print(f"  Saved {out}")


# Main
if __name__ == '__main__':
    print("Loading existing daily CSVs...")
    daily_data = load_existing_daily_csvs()

    if not daily_data:
        raise RuntimeError("No daily CSVs found. Run main.py first.")

    print(f"Loaded {len(daily_data)} (network, strategy) datasets.")

    print("Computing summary metrics...")
    summary = compute_summary(daily_data)

    summary_out = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    summary.to_csv(summary_out, index=False)
    print(f"Summary saved to {summary_out}")

    print("Generating plots...")
    plot_qeff_vs_time(daily_data)
    plot_qeff_vs_network_metrics(summary)

    print(f"Done. All plots saved to {PLOTS_DIR}/")