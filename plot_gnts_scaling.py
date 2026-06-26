"""
plot_gnts_scaling.py
--------------------
Plots LocalGNTS vs GlobalGNTS training time scaling against number of nodes.

Network -> node count mapping:
    SBM-1k  ->  1000   (exact)
    SBM-2k  ->  2000
    SBM-3k  ->  3000
    SBM-4k  ->  4000
    SBM-5k-* -> 5000   (average across all five SBM-5k variants)
    Orkut / LiveJournal / Youtube -> 10000 proxy (average across three SNAPs)

One figure per timing metric (4 total).
Each figure: LocalGNTS vs GlobalGNTS, x = node count, y = time metric.

Best metric for scaling analysis: seconds_per_iteration
(isolates pure per-step cost, free of startup overhead)

Usage:
    python plot_gnts_scaling.py
    python plot_gnts_scaling.py --csv gnts_model_training_times.csv
    python plot_gnts_scaling.py --out plots/
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
SNAP_NODE_COUNT = 10000   # proxy label for x-axis

# ---------------------------------------------------------------------------
# Models to compare
# ---------------------------------------------------------------------------
LOCAL_NAMES  = ["LocalGNTS"]    # extend if naming varies
GLOBAL_NAMES = ["GlobalGNTS"]

# ---------------------------------------------------------------------------
# Metrics to plot — (column, y-label, title, is_best)
# ---------------------------------------------------------------------------
METRICS = [
    ("seconds_per_iteration", "Seconds per Iteration (s)",
     "Seconds per Iteration vs Network Size", True),
    ("elapsed_seconds",       "Total Elapsed Seconds (s)",
     "Total Elapsed Seconds vs Network Size", False),
    ("wall_elapsed_seconds",  "Wall Elapsed Seconds (s)",
     "Wall Elapsed Seconds vs Network Size", False),
    ("elapsed_time",          "Elapsed Time (s, converted)",
     "Elapsed Time vs Network Size", False),
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
# Helpers
# ---------------------------------------------------------------------------

def parse_elapsed_time(series):
    """
    Convert HH:MM:SS or MM:SS strings to float seconds.
    Falls back to numeric coercion if already numeric.
    """
    def _convert(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        try:
            return float(s)
        except ValueError:
            pass
        parts = s.split(":")
        try:
            parts = [float(p) for p in parts]
            if len(parts) == 3:
                return parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                return parts[0] * 60 + parts[1]
        except ValueError:
            return np.nan
        return np.nan
    return series.apply(_convert)


def classify_model(name):
    """Return 'LocalGNTS', 'GlobalGNTS', or None."""
    n = str(name).strip()
    for local in LOCAL_NAMES:
        if local.lower() in n.lower():
            return "LocalGNTS"
    for glob in GLOBAL_NAMES:
        if glob.lower() in n.lower():
            return "GlobalGNTS"
    return None


def assign_node_count(network):
    """Map network name to node count. Returns None if unrecognised."""
    n = str(network).strip()
    if n in EXACT_NODE_COUNTS:
        return EXACT_NODE_COUNTS[n]
    if n in SBM5K_NETWORKS:
        return SBM5K_NODE_COUNT
    if n in SNAP_NETWORKS:
        return SNAP_NODE_COUNT
    return None


def build_scaling_df(raw_df):
    """
    From the raw CSV, compute per-model per-node-count mean for each metric.
    SBM-5k variants are averaged to 5000 nodes.
    SNAP networks are averaged to 10000 nodes.

    Returns a DataFrame with columns:
        model, node_count, <metric columns>
    """
    df = raw_df.copy()
    df.columns = df.columns.str.strip()

    # Convert elapsed_time to seconds if it's a string
    if "elapsed_time" in df.columns:
        df["elapsed_time"] = parse_elapsed_time(df["elapsed_time"])

    # Classify model
    df["_model"] = df["model"].apply(classify_model)
    df = df[df["_model"].notna()].copy()

    # Assign node count
    df["_node_count"] = df["network"].apply(assign_node_count)
    df = df[df["_node_count"].notna()].copy()

    metric_cols = [m[0] for m in METRICS if m[0] in df.columns]

    # Group by model + node_count and average (handles SBM-5k and SNAP groups)
    agg = (
        df.groupby(["_model", "_node_count"])[metric_cols]
        .mean()
        .reset_index()
        .rename(columns={"_model": "model", "_node_count": "node_count"})
        .sort_values(["model", "node_count"])
    )

    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_one_metric(agg_df, col, ylabel, title, out_dir, is_best=False):
    """One figure: LocalGNTS vs GlobalGNTS for a single metric."""
    if col not in agg_df.columns:
        print(f"  Column '{col}' not found — skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for model_name in ["LocalGNTS", "GlobalGNTS"]:
        sub = agg_df[agg_df["model"] == model_name].dropna(subset=[col])
        if sub.empty:
            continue
        ax.plot(
            sub["node_count"],
            sub[col],
            marker=MODEL_MARKERS[model_name],
            color=MODEL_COLOURS[model_name],
            label=model_name,
            linewidth=2.0,
            markersize=7,
        )
        # Annotate each point with its value
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row[col]:.2f}",
                xy=(row["node_count"], row[col]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=MODEL_COLOURS[model_name],
            )

    # x-axis ticks at the actual node counts present
    node_counts = sorted(agg_df["node_count"].unique())
    ax.set_xticks(node_counts)
    ax.set_xticklabels([f"{int(n/1000)}k" for n in node_counts])

    ax.set_xlabel("Network Size (nodes)")
    ax.set_ylabel(ylabel)

    best_tag = "*" if is_best else ""
    ax.set_title(f"{title}{best_tag}", fontsize=10)

    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    safe_col = col.replace(" ", "_")
    out_path = os.path.join(out_dir, f"scaling_{safe_col}.svg")
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="gnts_model_training_times.csv",
                        help="Path to CSV (default: gnts_model_training_times.csv)")
    parser.add_argument("--out", default="plots",
                        help="Output directory (default: plots/)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.csv} ...")
    raw_df = pd.read_csv(args.csv)
    print(f"  {len(raw_df)} rows, columns: {list(raw_df.columns)}")

    print("Building scaling summary ...")
    agg_df = build_scaling_df(raw_df)

    print("\nScaling summary (averaged):")
    print(agg_df.to_string(index=False))
    print()

    print("Generating plots ...")
    for col, ylabel, title, is_best in METRICS:
        plot_one_metric(agg_df, col, ylabel, title, args.out, is_best)

    print(f"\nDone. {len(METRICS)} plots saved to {args.out}/")
    print("Recommended plot for scaling analysis: plots/scaling_seconds_per_iteration.svg")


if __name__ == "__main__":
    main()
