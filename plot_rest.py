"""
plot_gnts_scaling.py
--------------------
Generates:
1. Scaling line plot for training times (seconds_per_iteration).
2. Summary CSV of all peak shift experiments.
3. Individual line plots of Detection Ratio per network per scenario.
4. A single Heatmap of Detection Ratio (Method vs Peak Day).

Usage:
    pip install seaborn
    python plot_gnts_scaling.py --csv gnts_model_training_times.csv --peak_csvs csvs_peak_shift/ --out plots/
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Network -> node count mapping
# ---------------------------------------------------------------------------
EXACT_NODE_COUNTS = {
    "SBM-1k":  1000,
    "SBM-2k":  2000,
    "SBM-3k":  3000,
    "SBM-4k":  4000,
}

SBM5K_NETWORKS = ["SBM-5k-Zero", "SBM-5k-Low", "SBM-5k-Med",
                  "SBM-5k-High", "SBM-5k-Max"]
SBM5K_NODE_COUNT = 5000

SNAP_NETWORKS = ["Orkut", "LiveJournal", "Youtube"]
SNAP_NODE_COUNT = 10000

# ---------------------------------------------------------------------------
# Models to compare
# ---------------------------------------------------------------------------
LOCAL_NAMES  = ["LocalGNTS"]
GLOBAL_NAMES = ["GlobalGNTS"]

METRICS = [
    ("seconds_per_iteration", "Seconds per Iteration (s)",
     "Seconds per Iteration vs Network Size"),
]

MODEL_COLOURS = {
    "LocalGNTS":  "#1f77b4",
    "GlobalGNTS": "#ff7f0e",
}
MODEL_MARKERS = {
    "LocalGNTS":  "o",
    "GlobalGNTS": "s",
}

# ---------------------------------------------------------------------------
# Helpers for Scaling
# ---------------------------------------------------------------------------
def parse_elapsed_time(series):
    def _convert(val):
        if pd.isna(val): return np.nan
        s = str(val).strip()
        try: return float(s)
        except ValueError: pass
        parts = s.split(":")
        try:
            parts = [float(p) for p in parts]
            if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2: return parts[0] * 60 + parts[1]
        except ValueError: return np.nan
        return np.nan
    return series.apply(_convert)

def classify_model(name):
    n = str(name).strip()
    for local in LOCAL_NAMES:
        if local.lower() in n.lower(): return "LocalGNTS"
    for glob in GLOBAL_NAMES:
        if glob.lower() in n.lower(): return "GlobalGNTS"
    return None

def assign_node_count(network):
    n = str(network).strip()
    if n in EXACT_NODE_COUNTS: return EXACT_NODE_COUNTS[n]
    if n in SBM5K_NETWORKS: return SBM5K_NODE_COUNT
    if n in SNAP_NETWORKS: return SNAP_NODE_COUNT
    return None

def build_scaling_df(raw_df):
    df = raw_df.copy()
    df.columns = df.columns.str.strip()
    if "elapsed_time" in df.columns: df["elapsed_time"] = parse_elapsed_time(df["elapsed_time"])
    df["_model"] = df["model"].apply(classify_model)
    df = df[df["_model"].notna()].copy()
    df["_node_count"] = df["network"].apply(assign_node_count)
    df = df[df["_node_count"].notna()].copy()
    metric_cols = [m[0] for m in METRICS if m[0] in df.columns]
    agg = (df.groupby(["_model", "_node_count"])[metric_cols].mean().reset_index()
           .rename(columns={"_model": "model", "_node_count": "node_count"})
           .sort_values(["model", "node_count"]))
    return agg

def plot_one_metric(agg_df, col, ylabel, title, out_dir):
    if col not in agg_df.columns: return
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name in ["LocalGNTS", "GlobalGNTS"]:
        sub = agg_df[agg_df["model"] == model_name].dropna(subset=[col])
        if sub.empty: continue
        ax.plot(sub["node_count"], sub[col], marker=MODEL_MARKERS[model_name],
                color=MODEL_COLOURS[model_name], label=model_name, linewidth=2.0, markersize=7)
    node_counts = sorted(agg_df["node_count"].unique())
    ax.set_xticks(node_counts)
    ax.set_xticklabels([f"{int(n/1000)}k" for n in node_counts])
    ax.set_xlabel("Network Size (nodes)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.savefig(os.path.join(out_dir, f"scaling_{col.replace(' ', '_')}.svg"), format="svg", bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Peak Shift Processing (CSV, Line Plots, Heatmaps)
# ---------------------------------------------------------------------------
def process_peak_shift_experiments(csv_dir, out_dir):
    if not os.path.exists(csv_dir):
        print(f"  Peak shift CSV directory '{csv_dir}' not found. Skipping.")
        return

    scenarios = ["Peak_25", "Peak_40", "Peak_60", "Peak_80", "Peak_100"]
    strategies = ["LocalGNTS_14", "GlobalGNTS_14", "Beta_Binomial_14",
                  "Proportional","Uniform", "Random"]
    networks = ["SBM-5k-Med"] # Locked to the single network used in the updated experiment

    print("\nProcessing Peak Shift Data (CSVs and Line Plots)...")
    summary_records = []

    # 1. Generate individual line plots and aggregate summary data
    for scenario in scenarios:
        target_peak = int(scenario.split("_")[1])
        
        for net in networks:
            fig, ax = plt.subplots(figsize=(8, 5))
            plotted_anything = False
            node_count = assign_node_count(net)

            for strategy in strategies:
                filename = f"{net}_{scenario}_{strategy}.csv"
                filepath = os.path.join(csv_dir, filename)
                
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    
                    if "Efficiency_avg" in df.columns and "Day" in df.columns:
                        # Summary Extraction
                        detection_ratio = df["Efficiency_avg"].mean()
                        
                        # Find the actual peak day based on highest I_avg
                        if "I_avg" in df.columns:
                            actual_peak_day = df.loc[df["I_avg"].idxmax(), "Day"]
                        else:
                            actual_peak_day = target_peak

                        summary_records.append({
                            "expt_name": scenario,
                            "network_name": net,
                            "num_nodes": node_count,
                            "infection_peak": actual_peak_day,
                            "target_peak_label": target_peak, # Kept for heatmap alignment
                            "model": strategy,
                            "detection_ratio": detection_ratio
                        })

                        # Line Plotting
                        df = df.sort_values("Day")
                        clean_strategy_name = strategy.replace("_", "-")
                        ax.plot(df["Day"], df["Efficiency_avg"], label=clean_strategy_name, linewidth=1.5)
                        plotted_anything = True

            if plotted_anything:
                ax.set_xlabel("Simulation Day")
                ax.set_ylabel("Detection Ratio (Efficiency)")
                ax.set_title(f"Detection Ratio: {net} ({scenario.replace('_', ' ')})")
                ax.legend(fontsize=9)
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
                
                plot_filename = f"detection_ratio_{net}_{scenario.lower()}.svg"
                fig.savefig(os.path.join(out_dir, plot_filename), format="svg", bbox_inches="tight")
            
            plt.close(fig)

    # 2. Export Summary CSV
    if not summary_records:
        print("  No peak shift data found to summarize.")
        return

    summary_df = pd.DataFrame(summary_records)
    csv_out_path = os.path.join(out_dir, "summary_peak_shift.csv")
    
    # Reorder columns for the final CSV
    export_df = summary_df[["expt_name", "network_name", "infection_peak", "model", "detection_ratio"]]
    export_df.to_csv(csv_out_path, index=False)
    print(f"  Saved Summary CSV: {csv_out_path}")

    # 3. Generate Single Heatmap (Method vs Peak)
    print("Generating Detection Ratio Heatmap...")
    if not summary_df.empty:
        # Pivot the data: X = model, Y = target_peak_label, Value = detection_ratio
        # Use pivot_table with 'mean' to gracefully handle any potential duplicate rows
        pivot_df = summary_df.pivot_table(index="target_peak_label", columns="model", values="detection_ratio", aggfunc='mean')
        
        # Sort Y-axis descending so the latest peak (100) is at the top
        pivot_df = pivot_df.sort_index(ascending=False)
        
        # Order X-axis to keep GNTS together and baselines together
        strategy_order = ["LocalGNTS_14", "GlobalGNTS_14", "Beta_Binomial_14", "Proportional", "Uniform", "Random"]
        valid_cols = [c for c in strategy_order if c in pivot_df.columns]
        other_cols = [c for c in pivot_df.columns if c not in valid_cols]
        pivot_df = pivot_df[valid_cols + other_cols]
        
        # Plot Heatmap (Slightly wider to fit the method names)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="viridis", ax=ax, 
                    cbar_kws={'label': 'Mean Detection Ratio'})
        
        ax.set_xlabel("Allocation Strategy")
        ax.set_ylabel("Target Peak Infection Day")
        ax.set_title("Detection Ratio: GNTS vs Baselines across Epidemic Speeds")
        
        # Clean up x-axis labels (replace underscores, rotate)
        labels = [item.get_text().replace('_', '-') for item in ax.get_xticklabels()]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        heatmap_path = os.path.join(out_dir, "heatmap_methods_vs_peak.svg")
        fig.savefig(heatmap_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved Heatmap: {heatmap_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="csvs/gnts_model_training_times.csv",
                        help="Path to scaling CSV (default: gnts_model_training_times.csv)")
    parser.add_argument("--peak_csvs", default="csvs",
                        help="Path to peak shift CSV dir (default: csvs_peak_shift)")
    parser.add_argument("--out", default="plots",
                        help="Output directory (default: plots/)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. Scaling Plot
    if os.path.exists(args.csv):
        print(f"Loading {args.csv} ...")
        raw_df = pd.read_csv(args.csv)
        agg_df = build_scaling_df(raw_df)
        for col, ylabel, title in METRICS:
            plot_one_metric(agg_df, col, ylabel, title, args.out)
        print(f"  Saved scaling plots to {args.out}/")
    else:
        print(f"Scaling CSV '{args.csv}' not found. Skipping scaling plots.")

    # 2. Peak Shift Processing
    process_peak_shift_experiments(args.peak_csvs, args.out)

    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()