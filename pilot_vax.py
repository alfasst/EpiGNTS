# pilot_vax.py
# ---------------------------------------------------------------------------
# No-vaccination epidemic pilot for the vaccine workflow.
# Runs the epidemic on all 5k SBM networks under three candidate parameter
# scenarios WITHOUT any vaccination, then plots the infectious curve per
# network so you can visually confirm peak timing before committing to a
# full experiment run.
#
# Usage:
#   python pilot_vax.py
#
# Output:
#   plots_vax/pilot_<network>_<scenario>.svg  — one per (network, scenario)
#   plots_vax/pilot_summary.svg               — all networks + scenarios overlay
#   pilot_vax_results.csv                     — peak day + peak size per run
#
# No models are trained. No doses are administered.
# Each (network, scenario) is run PILOT_RUNS times and the infectious
# curves are averaged so stochastic noise does not mislead the calibration.
# ---------------------------------------------------------------------------

import os
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

import config
from netepi_vax import (
    S, E, I, A, V, R,
    build_sim_graph,
    make_initial_states_vax,
    run_seiqr_vax_step,
)

# ---------------------------------------------------------------------------
# Pilot settings
# ---------------------------------------------------------------------------
PILOT_RUNS   = 10       # runs per (network, scenario) — enough to average out noise
PLOTS_DIR    = "plots_vax"
RESULTS_FILE = "pilot_vax_results.csv"

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameter scenarios to pilot
# Shared params (GAMMA, ASYMPTOMATIC_PROB, WANING_IMMUNITY_PROB, HUB_*)
# are taken from config — only the four levers vary across scenarios.
# ---------------------------------------------------------------------------
SCENARIOS = {
    "A-conservative": {
        "VAX_BETA":                      0.008,
        "VAX_SIGMA":                     1 / 10.0,
        "VAX_INITIAL_INFECTED":          3,
        "VAX_LONG_RANGE_INFECTION_PROB": 0.0005,
    },
    "B-moderate": {
        "VAX_BETA":                      0.005,
        "VAX_SIGMA":                     1 / 10.0,
        "VAX_INITIAL_INFECTED":          3,
        "VAX_LONG_RANGE_INFECTION_PROB": 0.0002,
    },
    "C-aggressive": {
        "VAX_BETA":                      0.003,
        "VAX_SIGMA":                     1 / 12.0,
        "VAX_INITIAL_INFECTED":          3,
        "VAX_LONG_RANGE_INFECTION_PROB": 0.0001,
    },
}

# Target window — used to draw a shaded band on plots
PEAK_TARGET_MIN = 45
PEAK_TARGET_MAX = 55

# Colour per scenario
SCENARIO_COLOURS = {
    "A-conservative": "#1f77b4",
    "B-moderate":     "#ff7f0e",
    "C-aggressive":   "#2ca02c",
}


# ---------------------------------------------------------------------------
# Single no-vaccination epidemic run
# ---------------------------------------------------------------------------

def run_no_vax_episode(sim_graph, params):
    """
    Run one epidemic episode with NO vaccination.
    Returns a (SIMULATION_DAYS,) array of daily infectious counts (I + A).

    Parameters
    ----------
    sim_graph : dict  — precomputed sim structures
    params    : dict  — scenario parameter dict (VAX_BETA, VAX_SIGMA, etc.)
    """
    states = make_initial_states_vax(sim_graph, params["VAX_INITIAL_INFECTED"])
    curve  = np.zeros(config.SIMULATION_DAYS, dtype=float)

    for day in range(config.SIMULATION_DAYS):
        infectious   = int(np.sum((states == I) | (states == A)))
        curve[day]   = infectious

        run_seiqr_vax_step(
            states, sim_graph,
            beta                = params["VAX_BETA"],
            sigma               = params["VAX_SIGMA"],
            gamma               = config.GAMMA,
            asymptomatic_prob   = config.ASYMPTOMATIC_PROB,
            hub_id              = config.HUB_BLOCK_ID,
            hub_multiplier      = config.HUB_BETA_MULTIPLIER,
            long_range_prob     = params["VAX_LONG_RANGE_INFECTION_PROB"],
            waning_immunity_prob  = config.WANING_IMMUNITY_PROB,
            waning_vaccine_prob   = 0.0,    # no vaccination — waning irrelevant
        )

    return curve


# ---------------------------------------------------------------------------
# SBM graph builder (mirrors main_vax.py)
# ---------------------------------------------------------------------------

def _build_prob_matrix(net_config):
    n_blocks = len(net_config["block_sizes"])
    p_in     = net_config["p_in"]
    p_out    = net_config["p_out"]
    return [
        [p_in if i == j else p_out for j in range(n_blocks)]
        for i in range(n_blocks)
    ]


def build_sbm_graph(net_config):
    G_nx = nx.stochastic_block_model(
        net_config["block_sizes"],
        _build_prob_matrix(net_config),
        seed=42
    )
    for block_id, size in enumerate(net_config["block_sizes"]):
        start = sum(net_config["block_sizes"][:block_id])
        for node in range(start, start + size):
            G_nx.nodes[node]['block_id'] = block_id
    return G_nx


# ---------------------------------------------------------------------------
# Per-network pilot: one plot per network showing all scenarios
# ---------------------------------------------------------------------------

def run_network_pilot(net_config, sim_graph):
    """
    Run PILOT_RUNS episodes for each scenario on one network.
    Returns dict: scenario_name -> (mean_curve, peak_day, peak_size).
    """
    net_name = net_config["name"]
    results  = {}

    for scenario_name, params in SCENARIOS.items():
        print(f"    Scenario {scenario_name} ...")
        curves = []
        for _ in range(PILOT_RUNS):
            curve = run_no_vax_episode(sim_graph, params)
            curves.append(curve)

        curves     = np.array(curves)           # (PILOT_RUNS, SIMULATION_DAYS)
        mean_curve = curves.mean(axis=0)
        std_curve  = curves.std(axis=0)
        peak_day   = int(np.argmax(mean_curve))
        peak_size  = float(mean_curve[peak_day])

        results[scenario_name] = {
            "mean_curve": mean_curve,
            "std_curve":  std_curve,
            "peak_day":   peak_day,
            "peak_size":  peak_size,
        }

        print(f"      Peak: day {peak_day}  |  size {peak_size:.1f}")

    return results


# ---------------------------------------------------------------------------
# Plotting — per network
# ---------------------------------------------------------------------------

def plot_network_pilot(net_name, pilot_results):
    """
    One SVG per network.
    Shows mean infectious curve ± 1 std for each scenario.
    Target peak window shaded in green.
    """
    days = np.arange(config.SIMULATION_DAYS)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Target window
    ax.axvspan(PEAK_TARGET_MIN, PEAK_TARGET_MAX,
               alpha=0.12, color='green', label=f'Target peak window (day {PEAK_TARGET_MIN}–{PEAK_TARGET_MAX})')

    for scenario_name, res in pilot_results.items():
        colour     = SCENARIO_COLOURS[scenario_name]
        mean_curve = res["mean_curve"]
        std_curve  = res["std_curve"]
        peak_day   = res["peak_day"]
        peak_size  = res["peak_size"]

        params_str = (
            f"β={SCENARIOS[scenario_name]['VAX_BETA']}  "
            f"σ=1/{round(1/SCENARIOS[scenario_name]['VAX_SIGMA'])}  "
            f"lr={SCENARIOS[scenario_name]['VAX_LONG_RANGE_INFECTION_PROB']}"
        )
        label = f"{scenario_name}  [{params_str}]  peak=day {peak_day}"

        ax.plot(days, mean_curve, color=colour, label=label, linewidth=1.8)
        ax.fill_between(days,
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        color=colour, alpha=0.15)

        # Mark peak
        ax.axvline(peak_day, color=colour, linestyle=':', linewidth=0.9)
        ax.annotate(f"day {peak_day}",
                    xy=(peak_day, peak_size),
                    xytext=(peak_day + 1, peak_size * 1.02),
                    fontsize=7, color=colour)

    ax.axvline(config.VACCINATION_START_DAY, color='black',
               linestyle='--', linewidth=1.0, label=f'Vaccination start (day {config.VACCINATION_START_DAY})')

    ax.set_title(f"Pilot: No-Vaccination Epidemic — {net_name}\n"
                 f"({PILOT_RUNS} runs per scenario, mean ± 1 std)")
    ax.set_xlabel("Day")
    ax.set_ylabel("I + A (infectious)")
    ax.legend(fontsize=7, loc='upper right')

    out = os.path.join(PLOTS_DIR, f"pilot_{net_name}.svg")
    fig.savefig(out, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved {out}")


# ---------------------------------------------------------------------------
# Summary plot — all networks × scenarios on one figure (subplots)
# ---------------------------------------------------------------------------

def plot_summary(all_pilot_results):
    """
    Grid of subplots: one row per network, one line per scenario.
    Gives an at-a-glance view of peak timing across all networks.
    """
    networks = list(all_pilot_results.keys())
    n_nets   = len(networks)
    days     = np.arange(config.SIMULATION_DAYS)

    fig, axes = plt.subplots(n_nets, 1,
                             figsize=(12, 4 * n_nets),
                             sharex=True)
    if n_nets == 1:
        axes = [axes]

    for ax, net_name in zip(axes, networks):
        ax.axvspan(PEAK_TARGET_MIN, PEAK_TARGET_MAX,
                   alpha=0.12, color='green')
        ax.axvline(config.VACCINATION_START_DAY, color='black',
                   linestyle='--', linewidth=0.8)

        for scenario_name, res in all_pilot_results[net_name].items():
            colour = SCENARIO_COLOURS[scenario_name]
            ax.plot(days, res["mean_curve"],
                    color=colour, label=f"{scenario_name} (peak day {res['peak_day']})",
                    linewidth=1.6)
            ax.fill_between(days,
                            res["mean_curve"] - res["std_curve"],
                            res["mean_curve"] + res["std_curve"],
                            color=colour, alpha=0.12)

        ax.set_title(net_name, fontsize=9)
        ax.set_ylabel("I + A")
        ax.legend(fontsize=7)

    axes[-1].set_xlabel("Day")
    fig.suptitle("Pilot Summary: No-Vaccination Epidemic Across 5k SBM Networks\n"
                 "Green band = target peak window (day 45–55)",
                 fontsize=10)

    out = os.path.join(PLOTS_DIR, "pilot_summary.svg")
    fig.savefig(out, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved summary: {out}")


# ---------------------------------------------------------------------------
# CSV export — peak day + size per (network, scenario, run)
# ---------------------------------------------------------------------------

def export_pilot_csv(all_pilot_results):
    """
    Write peak day and peak size for each (network, scenario) to CSV.
    Useful for quickly comparing scenarios numerically.
    """
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Network", "Scenario",
            "VAX_BETA", "VAX_SIGMA", "VAX_INITIAL_INFECTED",
            "VAX_LONG_RANGE_INFECTION_PROB",
            "Peak_Day_Mean", "Peak_Size_Mean",
            "In_Target_Window",
        ])
        for net_name, pilot_results in all_pilot_results.items():
            for scenario_name, res in pilot_results.items():
                params     = SCENARIOS[scenario_name]
                in_window  = PEAK_TARGET_MIN <= res["peak_day"] <= PEAK_TARGET_MAX
                writer.writerow([
                    net_name,
                    scenario_name,
                    params["VAX_BETA"],
                    round(params["VAX_SIGMA"], 5),
                    params["VAX_INITIAL_INFECTED"],
                    params["VAX_LONG_RANGE_INFECTION_PROB"],
                    res["peak_day"],
                    round(res["peak_size"], 2),
                    "YES" if in_window else "NO",
                ])
    print(f"  Results saved to {RESULTS_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    all_pilot_results = {}

    for net_config in config.SBM_5K_NETWORKS:
        net_name = net_config["name"]
        print(f"\n{'='*55}")
        print(f"Network: {net_name}")
        print(f"{'='*55}")

        G_nx      = build_sbm_graph(net_config)
        sim_graph = build_sim_graph(G_nx)

        pilot_results = run_network_pilot(net_config, sim_graph)
        all_pilot_results[net_name] = pilot_results

        plot_network_pilot(net_name, pilot_results)

    print("\nGenerating summary plot...")
    plot_summary(all_pilot_results)

    print("\nExporting CSV...")
    export_pilot_csv(all_pilot_results)

    print("\n" + "="*55)
    print("Pilot complete. Recommended next steps:")
    print("  1. Open plots_vax/pilot_summary.svg")
    print("  2. Find the scenario whose peak falls in the green band (day 45–55)")
    print("  3. Copy those four VAX_* values into config.py")
    print("  4. Run main_vax.py")
    print("="*55)
