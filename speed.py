# compare_epidemic.py
# --------------------------------------------------
# Compares old (NetworkX node-attribute) vs new (vectorised NumPy)
# epidemic implementations across SBM-1k through SBM-4k.
#
# For a fair comparison both implementations run the PURE epidemic
# (no testing, no allocation strategy) so the only thing being
# measured is the speed and fidelity of run_seiqr_step itself.
#
# Outputs:
#   results/compare_timing.csv        — per-network per-run wall times
#   results/compare_curves_<NET>.png  — S/E/I/A/Q/R vs time, old vs new
# --------------------------------------------------

import time
import copy
import random
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import scipy.sparse as sp

import config
import network_epidemic as new_ne
import network_epidemic_old as old_ne

# --------------------------------------------------
# Configuration
# --------------------------------------------------
N_RUNS       = 10
DAYS         = config.SIMULATION_DAYS
COMPARTMENTS = ['S', 'E', 'I', 'A', 'Q', 'R']

TARGET_NETS = ['SBM-1k', 'SBM-2k', 'SBM-3k', 'SBM-4k']
NET_CONFIGS = {n['name']: n for n in config.TEST_NETWORKS if n['name'] in TARGET_NETS}

import os
os.makedirs("results", exist_ok=True)

# --------------------------------------------------
# SBM graph builder (self-contained, no dependency on netgen.py)
# --------------------------------------------------

def build_sbm(net_cfg):
    sizes = net_cfg['block_sizes']
    p_in, p_out = net_cfg['p_in'], net_cfg['p_out']
    prob = [[p_out] * len(sizes) for _ in sizes]
    for i in range(len(sizes)):
        prob[i][i] = p_in
    G = nx.stochastic_block_model(sizes, prob, seed=42)
    idx = 0
    for bid, size in enumerate(sizes):
        for _ in range(size):
            G.nodes[idx]['block_id'] = bid
            idx += 1
    return G


# --------------------------------------------------
# OLD: pure epidemic run (no testing, no strategy)
# --------------------------------------------------

def run_old(G_template):
    """Run pure SEAIRQ on a deep-copied graph using old node-attribute loop."""
    G = copy.deepcopy(G_template)
    old_ne.initialize_epidemic(G, config.INITIAL_INFECTED)

    records = []
    t0 = time.perf_counter()

    for day in range(DAYS):
        old_ne.run_seiqr_step(
            G,
            config.BETA,
            config.SIGMA,
            config.GAMMA,
            config.ASYMPTOMATIC_PROB,
            config.HUB_BLOCK_ID,
            config.HUB_BETA_MULTIPLIER,
            config.LONG_RANGE_INFECTION_PROB,
            config.WANING_IMMUNITY_PROB,
        )
        counts = {s: 0 for s in COMPARTMENTS}
        for _, d in G.nodes(data=True):
            st = d.get('state', 'S')
            if st in counts:
                counts[st] += 1
        records.append({'Day': day, **counts})

    elapsed = time.perf_counter() - t0
    return pd.DataFrame(records), elapsed


# --------------------------------------------------
# NEW: pure epidemic run (no testing, no strategy)
# --------------------------------------------------

def run_new(G_template, sim_structures):
    """Run pure SEAIRQ on a fresh state array using vectorised new code."""
    node_index, nodes, adj_csr, block_arr = sim_structures
    N = len(nodes)

    state_arr = new_ne.initialize_epidemic(N, config.INITIAL_INFECTED)

    records = []
    t0 = time.perf_counter()

    for day in range(DAYS):
        new_ne.run_seiqr_step(
            state_arr,
            adj_csr,
            block_arr,
            config.BETA,
            config.SIGMA,
            config.GAMMA,
            config.ASYMPTOMATIC_PROB,
            config.HUB_BLOCK_ID,
            config.HUB_BETA_MULTIPLIER,
            config.LONG_RANGE_INFECTION_PROB,
            config.WANING_IMMUNITY_PROB,
        )
        c = np.bincount(state_arr.astype(np.int64), minlength=6)
        records.append({
            'Day': day,
            'S': int(c[new_ne.S]),
            'E': int(c[new_ne.E]),
            'I': int(c[new_ne.I]),
            'A': int(c[new_ne.A]),
            'Q': int(c[new_ne.Q]),
            'R': int(c[new_ne.R]),
        })

    elapsed = time.perf_counter() - t0
    return pd.DataFrame(records), elapsed


# --------------------------------------------------
# Main loop
# --------------------------------------------------

timing_rows = []

for net_name in TARGET_NETS:
    print(f"\n{'='*50}")
    print(f"Network: {net_name}")
    print(f"{'='*50}")

    G = build_sbm(NET_CONFIGS[net_name])
    N = G.number_of_nodes()

    # Pre-compute new structures once — reused across all N_RUNS
    sim_structures = new_ne.build_sim_structures(G)

    old_run_dfs = []
    new_run_dfs = []
    old_times   = []
    new_times   = []

    for run in range(N_RUNS):
        print(f"  Run {run + 1}/{N_RUNS}", end='\r')

        df_old, t_old = run_old(G)
        df_new, t_new = run_new(G, sim_structures)

        old_run_dfs.append(df_old)
        new_run_dfs.append(df_new)
        old_times.append(t_old)
        new_times.append(t_new)

        timing_rows.append({
            'Network':  net_name,
            'N_nodes':  N,
            'Run':      run + 1,
            'Old_sec':  round(t_old, 4),
            'New_sec':  round(t_new, 4),
            'Speedup':  round(t_old / t_new, 2) if t_new > 0 else np.nan,
        })

    print(f"  Done. Old avg: {np.mean(old_times):.3f}s  "
          f"New avg: {np.mean(new_times):.3f}s  "
          f"Speedup: {np.mean(old_times)/np.mean(new_times):.1f}x")

    # Average curves across runs
    old_mean = pd.concat(old_run_dfs).groupby('Day').mean().reset_index()
    new_mean = pd.concat(new_run_dfs).groupby('Day').mean().reset_index()

    # --------------------------------------------------
    # Plot: 6-compartment comparison  (old vs new)
    # --------------------------------------------------
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"{net_name}  (N={N})  —  Old vs New epidemic dynamics "
        f"(mean of {N_RUNS} runs)",
        fontsize=13, y=1.01
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = {
        'S': '#4878CF', 'E': '#F5A623',
        'I': '#D0021B', 'A': '#9B59B6',
        'Q': '#1DB954', 'R': '#7F8C8D',
    }

    for idx, comp in enumerate(COMPARTMENTS):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        days = old_mean['Day'].values

        ax.plot(days, old_mean[comp].values,
                color=colors[comp], lw=2,   label='Old',  linestyle='--')
        ax.plot(days, new_mean[comp].values,
                color=colors[comp], lw=1.5, label='New',  linestyle='-',  alpha=0.8)

        # Light band: min–max envelope across runs
        old_stack = np.stack([df[comp].values for df in old_run_dfs])
        new_stack = np.stack([df[comp].values for df in new_run_dfs])

        ax.fill_between(days,
                        old_stack.min(axis=0), old_stack.max(axis=0),
                        color=colors[comp], alpha=0.10, label='_')
        ax.fill_between(days,
                        new_stack.min(axis=0), new_stack.max(axis=0),
                        color=colors[comp], alpha=0.10, label='_')

        ax.set_title(f"Compartment {comp}", fontsize=11)
        ax.set_xlabel("Day", fontsize=9)
        ax.set_ylabel("Node count", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Timing annotation box
    t_old_mean = np.mean(old_times)
    t_new_mean = np.mean(new_times)
    speedup    = t_old_mean / t_new_mean if t_new_mean > 0 else np.nan
    fig.text(
        0.5, -0.02,
        f"Timing (mean over {N_RUNS} runs)  —  "
        f"Old: {t_old_mean:.3f}s   New: {t_new_mean:.3f}s   "
        f"Speedup: {speedup:.1f}×",
        ha='center', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='#aaaaaa')
    )

    out_png = f"results/compare_curves_{net_name}.png"
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out_png}")


# --------------------------------------------------
# Summary timing plot  (all 4 networks)
# --------------------------------------------------
timing_df = pd.DataFrame(timing_rows)
timing_df.to_csv("results/compare_timing.csv", index=False)
print(f"\nSaved results/compare_timing.csv")

summary = (
    timing_df
    .groupby('Network')[['Old_sec', 'New_sec', 'Speedup']]
    .agg(['mean', 'std'])
    .round(3)
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Old vs New — Wall-time comparison across SBM networks", fontsize=13)

net_order  = TARGET_NETS
x          = np.arange(len(net_order))
width      = 0.35

old_means  = [timing_df[timing_df.Network == n]['Old_sec'].mean() for n in net_order]
new_means  = [timing_df[timing_df.Network == n]['New_sec'].mean() for n in net_order]
old_stds   = [timing_df[timing_df.Network == n]['Old_sec'].std()  for n in net_order]
new_stds   = [timing_df[timing_df.Network == n]['New_sec'].std()  for n in net_order]
speedups   = [timing_df[timing_df.Network == n]['Speedup'].mean() for n in net_order]

# Bar chart
ax = axes[0]
ax.bar(x - width/2, old_means, width, yerr=old_stds,
       label='Old', color='#4878CF', alpha=0.85, capsize=4)
ax.bar(x + width/2, new_means, width, yerr=new_stds,
       label='New', color='#D0021B', alpha=0.85, capsize=4)
ax.set_xticks(x)
ax.set_xticklabels(net_order, fontsize=10)
ax.set_ylabel("Wall time (seconds)", fontsize=10)
ax.set_title("Mean runtime per simulation", fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Speedup line
ax2 = axes[1]
ax2.plot(net_order, speedups, 'o-', color='#1DB954', lw=2, markersize=8)
for i, (n, s) in enumerate(zip(net_order, speedups)):
    ax2.annotate(f"{s:.1f}×", (n, s),
                 textcoords="offset points", xytext=(0, 8),
                 ha='center', fontsize=10)
ax2.set_ylabel("Speedup (Old / New)", fontsize=10)
ax2.set_title("Speedup factor by network", fontsize=11)
ax2.set_ylim(0, max(speedups) * 1.3)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig("results/compare_timing_summary.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved results/compare_timing_summary.png")

# --------------------------------------------------
# Print summary table
# --------------------------------------------------
print("\n--- Timing Summary (mean ± std over 10 runs) ---")
print(f"{'Network':<12} {'Old (s)':<14} {'New (s)':<14} {'Speedup':<10}")
print("-" * 52)
for n in net_order:
    sub = timing_df[timing_df.Network == n]
    print(
        f"{n:<12} "
        f"{sub['Old_sec'].mean():.3f} ± {sub['Old_sec'].std():.3f}   "
        f"{sub['New_sec'].mean():.3f} ± {sub['New_sec'].std():.3f}   "
        f"{sub['Speedup'].mean():.1f}×"
    )