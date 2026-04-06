# speed.py
# --------------------------------------------------
# Compares old (NetworkX node-attribute) vs new (vectorised NumPy)
# epidemic implementations across SBM-1k through SBM-4k.
#
# Tests the pure epidemic only (no testing, no strategy) so the only
# thing being measured is run_seiqr_step speed and fidelity.
#
# Fidelity:
#   - Max absolute deviation between mean compartment curves
#   - Two-sample KS test on final-day distributions (p > 0.05 = PASS)
#
# Outputs:
#   plots/compare_timing.csv
#   plots/compare_timing_summary.png
#   plots/compare_curves_<NET>.png
#   plots/fidelity_report.csv
# --------------------------------------------------

import time
import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from scipy import stats

import config
import network_epidemic_old as old_ne
import network_epidemic     as new_ne

os.makedirs("plots", exist_ok=True)

# --------------------------------------------------
# Configuration
# --------------------------------------------------
N_RUNS       = 20
DAYS         = config.SIMULATION_DAYS
COMPARTMENTS = ['S', 'E', 'I', 'A', 'Q', 'R']
KS_ALPHA     = 0.05

TARGET_NETS  = ['SBM-1k', 'SBM-2k', 'SBM-3k', 'SBM-4k']
NET_CONFIGS  = {n['name']: n for n in config.SBM_NETWORKS
                if n['name'] in TARGET_NETS}

# --------------------------------------------------
# SBM builder
# --------------------------------------------------

def build_sbm(net_cfg):
    sizes       = net_cfg['block_sizes']
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
# Run helpers
# --------------------------------------------------

def run_old(G_template):
    G = copy.deepcopy(G_template)
    old_ne.initialize_epidemic(G, config.INITIAL_INFECTED)
    records = []
    t0 = time.perf_counter()
    for day in range(DAYS):
        old_ne.run_seiqr_step(
            G, config.BETA, config.SIGMA, config.GAMMA,
            config.ASYMPTOMATIC_PROB, config.HUB_BLOCK_ID,
            config.HUB_BETA_MULTIPLIER, config.LONG_RANGE_INFECTION_PROB,
            config.WANING_IMMUNITY_PROB,
        )
        counts = {s: 0 for s in COMPARTMENTS}
        for _, d in G.nodes(data=True):
            counts[d.get('state', 'S')] += 1
        records.append({'Day': day, **counts})
    return pd.DataFrame(records), time.perf_counter() - t0


def run_new(sim_graph):
    states = new_ne.make_initial_states(sim_graph, config.INITIAL_INFECTED)
    records = []
    t0 = time.perf_counter()
    for day in range(DAYS):
        new_ne.run_seiqr_step(
            states, sim_graph,
            config.BETA, config.SIGMA, config.GAMMA,
            config.ASYMPTOMATIC_PROB, config.HUB_BLOCK_ID,
            config.HUB_BETA_MULTIPLIER, config.LONG_RANGE_INFECTION_PROB,
            config.WANING_IMMUNITY_PROB,
        )
        c = np.bincount(states.astype(np.int64), minlength=6)
        records.append({
            'Day': day,
            'S': int(c[new_ne.S]), 'E': int(c[new_ne.E]),
            'I': int(c[new_ne.I]), 'A': int(c[new_ne.A]),
            'Q': int(c[new_ne.Q]), 'R': int(c[new_ne.R]),
        })
    return pd.DataFrame(records), time.perf_counter() - t0

# --------------------------------------------------
# Main loop
# --------------------------------------------------
timing_rows   = []
fidelity_rows = []

for net_name in TARGET_NETS:
    print(f"\n{'='*50}  {net_name}  {'='*50}")

    G         = build_sbm(NET_CONFIGS[net_name])
    N         = G.number_of_nodes()
    sim_graph = new_ne.build_sim_graph(G)   # built once, reused across all runs

    old_dfs, new_dfs     = [], []
    old_times, new_times = [], []

    for run in range(N_RUNS):
        print(f"  run {run+1}/{N_RUNS}", end='\r')
        df_old, t_old = run_old(G)
        df_new, t_new = run_new(sim_graph)
        old_dfs.append(df_old);   new_dfs.append(df_new)
        old_times.append(t_old);  new_times.append(t_new)
        timing_rows.append({
            'Network': net_name, 'N_nodes': N, 'Run': run + 1,
            'Old_sec': round(t_old, 4), 'New_sec': round(t_new, 4),
            'Speedup': round(t_old / t_new, 2) if t_new > 0 else np.nan,
        })

    speedup = np.mean(old_times) / np.mean(new_times)
    print(f"  Old: {np.mean(old_times):.3f}s  "
          f"New: {np.mean(new_times):.3f}s  "
          f"Speedup: {speedup:.1f}×")

    old_mean = pd.concat(old_dfs).groupby('Day').mean().reset_index()
    new_mean = pd.concat(new_dfs).groupby('Day').mean().reset_index()

    # Fidelity
    for comp in COMPARTMENTS:
        mad           = float(np.max(np.abs(old_mean[comp].values
                                            - new_mean[comp].values)))
        old_final     = np.array([df[comp].iloc[-1] for df in old_dfs])
        new_final     = np.array([df[comp].iloc[-1] for df in new_dfs])
        ks_stat, ks_p = stats.ks_2samp(old_final, new_final)
        fidelity_rows.append({
            'Network': net_name, 'Compartment': comp,
            'MaxAbsDev': round(mad, 2),
            'KS_stat': round(ks_stat, 4), 'KS_pval': round(ks_p, 4),
            'PASS': '✅' if ks_p > KS_ALPHA else '⚠️ ',
        })

    # Compartment curves plot
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"{net_name} (N={N}) — Old vs New epidemic dynamics "
                 f"(mean of {N_RUNS} runs)", fontsize=13, y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    colors = {'S': '#4878CF', 'E': '#F5A623', 'I': '#D0021B',
              'A': '#9B59B6', 'Q': '#1DB954', 'R': '#7F8C8D'}
    for idx, comp in enumerate(COMPARTMENTS):
        ax   = fig.add_subplot(gs[idx // 3, idx % 3])
        days = old_mean['Day'].values
        ax.plot(days, old_mean[comp].values, color=colors[comp],
                lw=2,   linestyle='--', label='Old')
        ax.plot(days, new_mean[comp].values, color=colors[comp],
                lw=1.5, linestyle='-',  label='New', alpha=0.8)
        old_stack = np.stack([df[comp].values for df in old_dfs])
        new_stack = np.stack([df[comp].values for df in new_dfs])
        ax.fill_between(days, old_stack.min(0), old_stack.max(0),
                        color=colors[comp], alpha=0.10)
        ax.fill_between(days, new_stack.min(0), new_stack.max(0),
                        color=colors[comp], alpha=0.10)
        ax.set_title(f"Compartment {comp}", fontsize=11)
        ax.set_xlabel("Day", fontsize=9); ax.set_ylabel("Nodes", fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.text(0.5, -0.02,
             f"Timing (mean {N_RUNS} runs) — "
             f"Old: {np.mean(old_times):.3f}s  "
             f"New: {np.mean(new_times):.3f}s  "
             f"Speedup: {speedup:.1f}×",
             ha='center', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0',
                       edgecolor='#aaaaaa'))
    out = f"plots/compare_curves_{net_name}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved {out}")

# --------------------------------------------------
# Fidelity report
# --------------------------------------------------
fid_df = pd.DataFrame(fidelity_rows)
print("\n" + "="*65)
print("FIDELITY REPORT")
print("="*65)
print(fid_df.to_string(index=False))
fid_df.to_csv("plots/fidelity_report.csv", index=False)
n_fail = (fid_df['PASS'] == '⚠️ ').sum()
n_pass = (fid_df['PASS'] == '✅').sum()
print(f"\nOverall: {n_pass} PASS  |  {n_fail} FAIL  (KS α={KS_ALPHA})")

# --------------------------------------------------
# Timing summary CSV + plots
# --------------------------------------------------
timing_df = pd.DataFrame(timing_rows)
timing_df.to_csv("plots/compare_timing.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Old vs New — Wall-time comparison across SBM networks", fontsize=13)

x        = np.arange(len(TARGET_NETS))
width    = 0.35
old_m    = [timing_df[timing_df.Network == n]['Old_sec'].mean() for n in TARGET_NETS]
new_m    = [timing_df[timing_df.Network == n]['New_sec'].mean() for n in TARGET_NETS]
old_s    = [timing_df[timing_df.Network == n]['Old_sec'].std()  for n in TARGET_NETS]
new_s    = [timing_df[timing_df.Network == n]['New_sec'].std()  for n in TARGET_NETS]
speedups = [timing_df[timing_df.Network == n]['Speedup'].mean() for n in TARGET_NETS]

ax = axes[0]
ax.bar(x - width/2, old_m, width, yerr=old_s, label='Old',
       color='#4878CF', alpha=0.85, capsize=4)
ax.bar(x + width/2, new_m, width, yerr=new_s, label='New',
       color='#D0021B', alpha=0.85, capsize=4)
ax.set_xticks(x); ax.set_xticklabels(TARGET_NETS, fontsize=10)
ax.set_ylabel("Wall time (seconds)", fontsize=10)
ax.set_title("Mean runtime per simulation", fontsize=11)
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

ax2 = axes[1]
ax2.plot(TARGET_NETS, speedups, 'o-', color='#1DB954', lw=2, markersize=8)
for n, s in zip(TARGET_NETS, speedups):
    ax2.annotate(f"{s:.1f}×", (n, s),
                 textcoords="offset points", xytext=(0, 8),
                 ha='center', fontsize=10)
ax2.set_ylabel("Speedup (Old / New)", fontsize=10)
ax2.set_title("Speedup factor by network", fontsize=11)
ax2.set_ylim(0, max(speedups) * 1.3)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig("plots/compare_timing_summary.png", dpi=150, bbox_inches='tight')
plt.close(fig)

# Console summary
print(f"\n--- Timing Summary (mean ± std over {N_RUNS} runs) ---")
print(f"{'Network':<12} {'Old (s)':<20} {'New (s)':<20} {'Speedup'}")
print("-" * 58)
for n in TARGET_NETS:
    sub = timing_df[timing_df.Network == n]
    print(f"{n:<12} "
          f"{sub['Old_sec'].mean():.3f} ± {sub['Old_sec'].std():.3f}   "
          f"{sub['New_sec'].mean():.3f} ± {sub['New_sec'].std():.3f}   "
          f"{sub['Speedup'].mean():.1f}×")

print("\nSaved: plots/compare_timing.csv  "
      "plots/compare_timing_summary.png  plots/fidelity_report.csv")