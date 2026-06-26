# analyze_vax.py
# Analysis for per-network–per-strategy daily CSVs from the vaccine workflow.
# ---------------------------------------------------------------------------
# Design mirrors analyze.py exactly:
# - Each (Network, Strategy) has its own daily CSV in csvs_vax/
# - analyze_vax.py DISCOVERS existing CSVs
# - Missing combinations are silently skipped
# - All summaries and plots are derived from daily data or summary_metrics_vax.csv
#
# Changes from previous version:
#   1. NETWORKS -> SBM_5K_NETWORKS (was SNAP_NETWORKS)
#   2. plot_dose_efficiency_bar() added (Plot 7)
#      reads ProductiveDoses / TotalDoses from summary_metrics_vax.csv
# ---------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR  = "csvs_vax"
SUMMARY_FILE = os.path.join(RESULTS_DIR, "summary_metrics_vax.csv")
PLOTS_DIR    = "plots_vax"

os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Canonical lists — 5k SBM networks only
# ---------------------------------------------------------------------------
NETWORKS = [n['name'] for n in config.SBM_5K_NETWORKS]

STRATEGIES = [
    "LocalGNTSVax-14",
    "GlobalGNTSVax-14",
    "BetaBinomialVax-14",
    "GammaPoissonVax-14",
    "Proportional",
    "RiskWeighted",
    "Uniform",
    "Random",
]

# Colour palette — one colour per strategy, consistent across all plots
STRATEGY_COLOURS = {
    "LocalGNTSVax-14":    "#1f77b4",
    "GlobalGNTSVax-14":   "#ff7f0e",
    "BetaBinomialVax-14": "#2ca02c",
    "GammaPoissonVax-14": "#d62728",
    "Proportional":       "#9467bd",
    "RiskWeighted":       "#8c564b",
    "Uniform":            "#e377c2",
    "Random":             "#7f7f7f",
}


def _strategy_to_filename(strat):
    """Match the naming convention used in main_vax.py."""
    return strat.replace("-", "_")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_existing_daily_csvs():
    """
    Load all existing (Network, Strategy) daily CSVs from csvs_vax/.
    Returns dict keyed by (network_name, strategy_name) -> pd.DataFrame.
    Missing files are silently skipped.
    """
    data = {}
    for net in NETWORKS:
        for strat in STRATEGIES:
            path = os.path.join(
                RESULTS_DIR, f"{net}_{_strategy_to_filename(strat)}.csv"
            )
            if os.path.exists(path):
                data[(net, strat)] = pd.read_csv(path)
    return data


def add_derived_columns(df):
    """
    Add computed columns used by both summary and plotting.

    CSV columns available (from simulation_vax.py export):
        Day, S_avg, E_avg, I_avg, A_avg, V_avg, R_avg,
        Infectious_avg, Immune_avg, VaxCoverage_avg

    Added columns:
        infectious   = I_avg + A_avg
        immune       = V_avg + R_avg
        vax_coverage = VaxCoverage_avg (aliased for clarity)
        unvaccinated = S_avg + E_avg + I_avg + A_avg
    """
    df = df.copy()
    df['infectious']   = df['I_avg'] + df['A_avg']
    df['immune']       = df['V_avg'] + df['R_avg']
    df['vax_coverage'] = df['VaxCoverage_avg']
    df['unvaccinated'] = df['S_avg'] + df['E_avg'] + df['I_avg'] + df['A_avg']
    return df


# ---------------------------------------------------------------------------
# Summary computation from daily CSVs
# ---------------------------------------------------------------------------

def compute_summary_from_daily(daily_data):
    """
    Compute (Network, Strategy) level scalar summary from daily data.
    Already averaged across runs by export_strategy_results_vax in main_vax.py.

    Columns:
        Network, Strategy,
        peak_infections, time_to_peak,
        final_vax_coverage, avg_infectious,
        integrated_infections, herd_immunity_day
    """
    records   = []
    threshold = getattr(config, 'HERD_IMMUNITY_THRESHOLD', 0.7)

    for (net, strat), df in daily_data.items():
        df = add_derived_columns(df)

        pop_day0 = (
            df.loc[0, 'S_avg'] + df.loc[0, 'E_avg']
            + df.loc[0, 'I_avg'] + df.loc[0, 'A_avg']
            + df.loc[0, 'V_avg'] + df.loc[0, 'R_avg']
        )

        peak_idx  = df['infectious'].idxmax()
        peak_val  = df['infectious'].max()
        t_peak    = int(df.loc[peak_idx, 'Day'])
        final_vax = float(df.iloc[-1]['vax_coverage'])
        integrated = float(df['infectious'].sum())

        if pop_day0 > 0:
            herd_rows = df[df['immune'] / pop_day0 >= threshold]
            herd_day  = int(herd_rows['Day'].iloc[0]) if not herd_rows.empty else -1
        else:
            herd_day = -1

        records.append({
            'Network':               net,
            'Strategy':              strat,
            'peak_infections':       round(peak_val, 2),
            'time_to_peak':          t_peak,
            'final_vax_coverage':    round(final_vax, 4),
            'avg_infectious':        round(df['infectious'].mean(), 4),
            'integrated_infections': round(integrated, 2),
            'herd_immunity_day':     herd_day,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _savefig(fig, filename):
    out = os.path.join(PLOTS_DIR, filename)
    fig.savefig(out, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


def _bar_chart(ax, strategies_present, values, title, ylabel,
               ylim=None, hline=None, hline_label=None):
    """
    Shared bar-chart renderer used by plots 4, 5, 6, 7.
    Keeps all four bar plots visually consistent.
    """
    colors = [STRATEGY_COLOURS.get(s, '#333333') for s in strategies_present]
    x      = np.arange(len(strategies_present))
    ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies_present, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if hline is not None:
        ax.axhline(hline, color='red', linestyle=':', linewidth=0.8,
                   label=hline_label)
        ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Plot 1 — Infectious over time (I+A) per network
# ---------------------------------------------------------------------------

def plot_infectious_over_time(daily_data):
    """One SVG per network. Y-axis: I_avg + A_avg."""
    for net in NETWORKS:
        fig, ax = plt.subplots()
        plotted = False
        for strat in STRATEGIES:
            key = (net, strat)
            if key not in daily_data:
                continue
            df = add_derived_columns(daily_data[key])
            ax.plot(df['Day'], df['infectious'],
                    label=strat, color=STRATEGY_COLOURS.get(strat))
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(f"Active Infections Over Time ({net})")
        ax.set_xlabel("Day")
        ax.set_ylabel("I + A (avg across runs)")
        ax.axvline(config.VACCINATION_START_DAY, color='grey',
                   linestyle='--', linewidth=0.8, label='Campaign start')
        ax.legend(fontsize=7)
        _savefig(fig, f"infectious_over_time_{net}.svg")


# ---------------------------------------------------------------------------
# Plot 2 — Vaccine coverage over time per network
# ---------------------------------------------------------------------------

def plot_vax_coverage_over_time(daily_data):
    """One SVG per network. Y-axis: V / total population."""
    for net in NETWORKS:
        fig, ax = plt.subplots()
        plotted = False
        for strat in STRATEGIES:
            key = (net, strat)
            if key not in daily_data:
                continue
            df = add_derived_columns(daily_data[key])
            ax.plot(df['Day'], df['vax_coverage'],
                    label=strat, color=STRATEGY_COLOURS.get(strat))
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_title(f"Vaccine Coverage Over Time ({net})")
        ax.set_xlabel("Day")
        ax.set_ylabel("V / Population (avg across runs)")
        ax.set_ylim(0, 1)
        ax.axvline(config.VACCINATION_START_DAY, color='grey',
                   linestyle='--', linewidth=0.8, label='Campaign start')
        ax.legend(fontsize=7)
        _savefig(fig, f"vax_coverage_over_time_{net}.svg")


# ---------------------------------------------------------------------------
# Plot 3 — Immune (V+R) over time per network
# ---------------------------------------------------------------------------

def plot_immune_over_time(daily_data):
    """
    One SVG per network. Y-axis: V_avg + R_avg.
    Herd immunity threshold shown as horizontal dashed line.
    """
    threshold = getattr(config, 'HERD_IMMUNITY_THRESHOLD', 0.7)

    for net in NETWORKS:
        fig, ax  = plt.subplots()
        plotted  = False
        pop_day0 = None

        for strat in STRATEGIES:
            key = (net, strat)
            if key not in daily_data:
                continue
            df = add_derived_columns(daily_data[key])
            ax.plot(df['Day'], df['immune'],
                    label=strat, color=STRATEGY_COLOURS.get(strat))
            plotted = True
            if pop_day0 is None:
                pop_day0 = (
                    df.loc[0, 'S_avg'] + df.loc[0, 'E_avg']
                    + df.loc[0, 'I_avg'] + df.loc[0, 'A_avg']
                    + df.loc[0, 'V_avg'] + df.loc[0, 'R_avg']
                )

        if not plotted:
            plt.close(fig)
            continue

        if pop_day0 and pop_day0 > 0:
            ax.axhline(threshold * pop_day0, color='red', linestyle=':',
                       linewidth=0.8,
                       label=f'Herd threshold ({int(threshold * 100)}%)')

        ax.set_title(f"Immune Population (V+R) Over Time ({net})")
        ax.set_xlabel("Day")
        ax.set_ylabel("V + R (avg across runs)")
        ax.axvline(config.VACCINATION_START_DAY, color='grey',
                   linestyle='--', linewidth=0.8)
        ax.legend(fontsize=7)
        _savefig(fig, f"immune_over_time_{net}.svg")


# ---------------------------------------------------------------------------
# Plot 4 — Peak infections per strategy (bar chart, one per network)
# ---------------------------------------------------------------------------

def plot_peak_infections_bar(summary):
    """Bar chart: x = strategy, y = peak_infections."""
    for net in NETWORKS:
        sub = summary[summary['Network'] == net]
        if sub.empty:
            continue
        strategies_present = [s for s in STRATEGIES if s in sub['Strategy'].values]
        values = [float(sub[sub['Strategy'] == s]['peak_infections'].iloc[0])
                  for s in strategies_present]
        fig, ax = plt.subplots()
        _bar_chart(ax, strategies_present, values,
                   title=f"Peak Infections by Strategy ({net})",
                   ylabel="Peak Infections (avg across runs)")
        _savefig(fig, f"peak_infections_bar_{net}.svg")


# ---------------------------------------------------------------------------
# Plot 5 — Herd immunity day per strategy (bar chart, one per network)
# ---------------------------------------------------------------------------

def plot_herd_immunity_day_bar(summary):
    """
    Bar chart: x = strategy, y = herd_immunity_day.
    Strategies that never reached herd immunity plotted at SIMULATION_DAYS.
    """
    for net in NETWORKS:
        sub = summary[summary['Network'] == net]
        if sub.empty:
            continue
        strategies_present = [s for s in STRATEGIES if s in sub['Strategy'].values]
        values = [
            float(sub[sub['Strategy'] == s]['herd_immunity_day'].iloc[0])
            if float(sub[sub['Strategy'] == s]['herd_immunity_day'].iloc[0]) >= 0
            else config.SIMULATION_DAYS
            for s in strategies_present
        ]
        fig, ax = plt.subplots()
        _bar_chart(ax, strategies_present, values,
                   title=f"Herd Immunity Day by Strategy ({net})",
                   ylabel="Day herd immunity reached",
                   hline=config.SIMULATION_DAYS,
                   hline_label="End of simulation (not reached)")
        _savefig(fig, f"herd_immunity_day_bar_{net}.svg")


# ---------------------------------------------------------------------------
# Plot 6 — Waste rate per strategy (from summary_metrics_vax.csv)
# ---------------------------------------------------------------------------

def plot_waste_rate_bar(summary_metrics_csv):
    """
    Bar chart: x = strategy, y = mean WasteRate across runs.
    WasteRate = WastedDoses / TotalDoses (computed in main_vax.py).
    Reads from summary_metrics_vax.csv — not available in daily CSVs.
    """
    if not os.path.exists(summary_metrics_csv):
        print(f"  Skipping waste-rate plots: {summary_metrics_csv} not found.")
        return

    df_full = pd.read_csv(summary_metrics_csv)

    for net in NETWORKS:
        sub = df_full[df_full['Network'] == net]
        if sub.empty:
            continue
        strategies_present = [s for s in STRATEGIES if s in sub['Strategy'].values]
        values = [float(sub[sub['Strategy'] == s]['WasteRate'].mean())
                  for s in strategies_present]
        fig, ax = plt.subplots()
        _bar_chart(ax, strategies_present, values,
                   title=f"Dose Waste Rate by Strategy ({net})",
                   ylabel="Mean Waste Rate (wasted / total doses)",
                   ylim=(0, 1))
        _savefig(fig, f"waste_rate_bar_{net}.svg")


# ---------------------------------------------------------------------------
# Plot 7 — Dose efficiency per strategy (from summary_metrics_vax.csv)
# ---------------------------------------------------------------------------

def plot_dose_efficiency_bar(summary_metrics_csv):
    """
    Bar chart: x = strategy, y = mean DoseEfficiency across runs.
    DoseEfficiency = ProductiveDoses / TotalDoses.

    Complement of WasteRate — shows the fraction of doses that successfully
    immunised an S or E node. Plotted separately from waste rate because:
      - waste rate and dose efficiency sum to 1.0, so both are informative
        only when shown together for direct comparison
      - dose efficiency is the primary positive metric; waste rate is the
        cost metric — keeping them as separate plots avoids visual confusion

    Reads ProductiveDoses and TotalDoses from summary_metrics_vax.csv.
    DoseEfficiency computed here rather than in main_vax.py so that the
    summary CSV stays minimal (raw counts only, no derived ratios).
    """
    if not os.path.exists(summary_metrics_csv):
        print(f"  Skipping dose-efficiency plots: {summary_metrics_csv} not found.")
        return

    df_full = pd.read_csv(summary_metrics_csv)

    # Compute per-row dose efficiency; guard against zero total doses
    df_full['DoseEfficiency'] = np.where(
        df_full['TotalDoses'] > 0,
        df_full['ProductiveDoses'] / df_full['TotalDoses'],
        0.0
    )

    for net in NETWORKS:
        sub = df_full[df_full['Network'] == net]
        if sub.empty:
            continue
        strategies_present = [s for s in STRATEGIES if s in sub['Strategy'].values]
        values = [float(sub[sub['Strategy'] == s]['DoseEfficiency'].mean())
                  for s in strategies_present]
        fig, ax = plt.subplots()
        _bar_chart(ax, strategies_present, values,
                   title=f"Dose Efficiency by Strategy ({net})",
                   ylabel="Mean Dose Efficiency (productive / total doses)",
                   ylim=(0, 1))
        _savefig(fig, f"dose_efficiency_bar_{net}.svg")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Loading existing daily vaccine CSVs...")
    daily_data = load_existing_daily_csvs()

    if not daily_data:
        raise RuntimeError(
            f"No daily CSVs found in {RESULTS_DIR}/. Run main_vax.py first."
        )
    print(f"Loaded {len(daily_data)} (network, strategy) datasets.")

    print("Computing summary from daily data...")
    summary = compute_summary_from_daily(daily_data)

    summary_daily_out = os.path.join(RESULTS_DIR, "summary_daily_derived_vax.csv")
    summary.to_csv(summary_daily_out, index=False)
    print(f"Summary saved to {summary_daily_out}")

    print("Generating plots...")
    plot_infectious_over_time(daily_data)
    plot_vax_coverage_over_time(daily_data)
    plot_immune_over_time(daily_data)
    plot_peak_infections_bar(summary)
    plot_herd_immunity_day_bar(summary)
    plot_waste_rate_bar(SUMMARY_FILE)
    plot_dose_efficiency_bar(SUMMARY_FILE)

    print(f"\nDone. All plots saved to {PLOTS_DIR}/")
