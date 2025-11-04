# analyze.py
# Description: Hybrid analysis script that combines a legacy summary file with new, detailed metrics.
# It generates a complete, unified suite of plots for all 13 networks.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import networkx as nx
import numpy as np
import ast # For safely evaluating string-formatted dictionaries
import matplotlib.ticker as mticker

import config

# --- 1. Configuration & Setup ---
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# --- NEW: Added path for the legacy summary file ---
LEGACY_SUMMARY_FILE = os.path.join(RESULTS_DIR, "sbm_summary_legacy.csv")

SBM_METRICS_FILE = os.path.join(RESULTS_DIR, "network_metrics.csv")
SNAP_METRICS_FILE = os.path.join(RESULTS_DIR, "snap_network_metrics.csv")
TESTING_START_DAY = 20

STRATEGY_PALETTE = {
    'GNTS': '#1f77b4', 'MAB-TS': '#ff7f0e', 'Proportional': '#d62728',
    'Uniform': '#9467bd', 'Random': '#8c564b'
}

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


# --- 2. Data Loading and Summary Calculation ---

def process_day_dict(day_dict_str):
    try:
        day_dict = ast.literal_eval(day_dict_str)
        if not isinstance(day_dict, dict): return pd.Series([np.nan, np.nan, np.nan])
        total_communities = len(day_dict)
        affected_days = [day for day in day_dict.values() if day != -1]
        num_affected = len(affected_days)
        prop_affected = num_affected / total_communities if total_communities > 0 else 0
        avg_time = np.mean(affected_days) if num_affected > 0 else np.nan
        return pd.Series([num_affected, prop_affected, avg_time])
    except (ValueError, SyntaxError):
        return pd.Series([np.nan, np.nan, np.nan])

def calculate_summary_from_raw_data(results_dir):
    """
    Generates a final summary by combining a legacy summary file with newly generated metrics.
    """
    print("Calculating summary statistics from all raw run data...")
    metric_files = glob.glob(os.path.join(results_dir, "*_metrics.csv"))
    
    sbm_map = {v: k for k, v in config.NETWORK_SHORT_NAMES.items()}
    snap_map = {net['short_name']: net['name'] for net in config.SNAP_NETWORKS}
    short_to_long_name = {**sbm_map, **snap_map}

    all_summaries = []

    # --- Part 1: Process new metrics files (for SNAP networks) ---
    if metric_files:
        all_metrics_dfs = []
        for f in metric_files:
            basename = os.path.basename(f).replace('_metrics.csv', '')
            parts = basename.split('_')
            strategy, network_short_name = parts[-1], '_'.join(parts[:-1])
            if network_short_name not in short_to_long_name:
                 strategy, network_short_name = '_'.join(parts[-2:]), '_'.join(parts[:-2])
            if network_short_name in short_to_long_name:
                df = pd.read_csv(f)
                df['Network_Name'] = short_to_long_name[network_short_name]
                df['Strategy'] = strategy
                all_metrics_dfs.append(df)
        
        if all_metrics_dfs:
            metrics_df = pd.concat(all_metrics_dfs)
            if 'first_infection_day' in metrics_df.columns:
                metrics = metrics_df['first_infection_day'].apply(process_day_dict)
                metrics.columns = ['communities_infected_count', 'proportion_infected', 'avg_time_to_infection']
                metrics_df = pd.concat([metrics_df, metrics], axis=1).drop(columns=['first_infection_day'])
            if 'first_intervention_day' in metrics_df.columns:
                metrics = metrics_df['first_intervention_day'].apply(process_day_dict)
                metrics.columns = ['communities_intervened_count', 'proportion_intervened', 'avg_time_to_intervention']
                metrics_df = pd.concat([metrics_df, metrics], axis=1).drop(columns=['first_intervention_day'])
            
            new_summary = metrics_df.groupby(['Network_Name', 'Strategy']).mean().reset_index()
            all_summaries.append(new_summary)

    # --- Part 2: Load legacy summary file (for SBM networks) ---
    if os.path.exists(LEGACY_SUMMARY_FILE):
        print(f"Loading legacy summary from '{LEGACY_SUMMARY_FILE}'...")
        legacy_summary_df = pd.read_csv(LEGACY_SUMMARY_FILE)
        all_summaries.append(legacy_summary_df)
    else:
        print(f"Warning: Legacy summary file not found at '{LEGACY_SUMMARY_FILE}'.")

    if not all_summaries:
        print("Error: No data found to generate summary.")
        return None, None

    # --- Part 3: Combine summaries and process raw daily data ---
    final_summary = pd.concat(all_summaries, ignore_index=True)
    final_summary.rename(columns={'peak_infections': 'Avg_Peak_Infections', 'time_to_peak': 'Avg_Time_to_Peak', 'total_new_infections': 'Avg_Total_Infections'}, inplace=True, errors='ignore')

    raw_daily_files = [f for f in glob.glob(os.path.join(results_dir, "*.csv")) if '_metrics.csv' not in f and 'summary' not in f and 'network_metrics' not in f]
    if raw_daily_files:
        all_daily_dfs = []
        for f in raw_daily_files:
            basename, strategy = os.path.basename(f).replace('.csv', '').split('_', 1)
            if basename in short_to_long_name:
                df = pd.read_csv(f)
                df['Network_Name'] = short_to_long_name[basename]
                df['Strategy'] = strategy
                all_daily_dfs.append(df)
        raw_df = pd.concat(all_daily_dfs) if all_daily_dfs else pd.DataFrame()
        
        # Calculate efficiency for new networks and merge it into the summary
        efficiency_summary = raw_df.groupby(['Network_Name', 'Strategy'])['Efficiency'].mean().reset_index().rename(columns={'Efficiency': 'Avg_Quarantine_Efficiency'})
        # Update summary for new networks; keep legacy values for old networks
        final_summary.set_index(['Network_Name', 'Strategy'], inplace=True)
        final_summary.update(efficiency_summary.set_index(['Network_Name', 'Strategy']))
        final_summary.reset_index(inplace=True)
    else:
        raw_df = pd.DataFrame() # Return empty dataframe if no raw files

    final_summary.to_csv(os.path.join(results_dir, "strategy_summary_COMBINED.csv"), index=False)
    print(f"✅ Combined summary data saved for {len(final_summary['Network_Name'].unique())} networks.")
    return final_summary, raw_df


# --- 3. Network Metric Calculation ---
# ... (This section remains unchanged, functions are not shown for brevity)
def calculate_and_save_sbm_metrics(output_path):
    print("Calculating structural metrics for all SBM networks...")
    metrics = []
    for net_params in config.TEST_NETWORKS:
        name, sizes, p_in, p_out = net_params["name"], net_params["block_sizes"], net_params["p_in"], net_params["p_out"]
        prob_matrix = [[p_out] * len(sizes) for _ in range(len(sizes))]
        for i in range(len(sizes)): prob_matrix[i][i] = p_in
        G = nx.stochastic_block_model(sizes, prob_matrix, seed=42)
        node_block_map = {node: data['block'] for node, data in G.nodes(data=True)}
        communities = [set(n for n, b in node_block_map.items() if b == i) for i in range(len(sizes))]
        modularity_score = nx.community.modularity(G, [c for c in communities if c])
        ccc = np.mean([np.mean([nx.clustering(G, n) for n in c]) for c in communities if c]) if communities else 0.0
        metrics.append({"Network_Name": name, "Modularity": modularity_score, "Clustering": ccc, "Size": sum(sizes)})
    pd.DataFrame(metrics).to_csv(output_path, index=False)
    print(f"✅ SBM structural metrics saved.")
    return pd.DataFrame(metrics)

# --- 4. Plotting Functions ---

def plot_performance_curves(df, output_dir):
    """Generates performance curves for networks with available raw data."""
    if df.empty:
        print("No raw daily data available to plot performance curves.")
        return
    print("\nGenerating performance curves for available networks...")
    for network in df['Network_Name'].unique():
        filename = os.path.join(output_dir, f"performance_curve_{network}.svg")
        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(data=df[df['Network_Name'] == network], x='Day', y='Efficiency', hue='Strategy', palette=STRATEGY_PALETTE, linewidth=2, errorbar=('ci', 95))
        ax.axvline(x=TESTING_START_DAY, color='black', linestyle='--', linewidth=1.5, label='Testing Starts')
        ax.set_title(f'Strategy Performance on {network}', fontsize=16, weight='bold')
        ax.set_xlabel('Simulation Day'); ax.set_ylabel('Avg. Quarantine Efficiency (with 95% CI)')
        ax.legend(title='Strategy'); ax.set_ylim(0)
        plt.tight_layout(); plt.savefig(filename, format='svg'); plt.close()
    print("✅ Performance curve plotting complete.")

def plot_efficiency_vs_size(df, output_dir):
    print("\nGenerating unified efficiency vs. size plot...")
    sbm_size_nets = ["Fragmented_1k", "Core-Periphery_2k", "Bimodal_3k", "Pyramidal_4k", "SBM_5k_Medium_Modularity"]
    snap_nets = [net['name'] for net in config.SNAP_NETWORKS]
    plot_df = df[df['Network_Name'].isin(sbm_size_nets + snap_nets)]
    if plot_df.empty: return

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=plot_df, x='Size', y='Avg_Quarantine_Efficiency', hue='Strategy', palette=STRATEGY_PALETTE, marker='o')
    ax.set_title('Strategy Efficiency vs. Network Scale', fontsize=16, weight='bold')
    ax.set_xlabel('Total Network Nodes (Log Scale)'); ax.set_ylabel('Average Quarantine Efficiency')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "unified_eff_vs_size.svg")); plt.close()
    print("✅ Unified size plot saved.")

def plot_efficiency_vs_modularity(df, output_dir):
    print("\nGenerating unified efficiency vs. modularity plot...")
    sbm_mod_nets = [net['name'] for net in config.TEST_NETWORKS if 'SBM_5k_' in net['name']]
    snap_nets = [net['name'] for net in config.SNAP_NETWORKS]
    plot_df = df[df['Network_Name'].isin(sbm_mod_nets + snap_nets)]
    if plot_df.empty or 'Modularity' not in plot_df.columns: return
    
    plot_df = plot_df.sort_values('Modularity')
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=plot_df, x='Modularity', y='Avg_Quarantine_Efficiency', hue='Strategy', palette=STRATEGY_PALETTE, marker='o')
    ax.set_title('Strategy Efficiency vs. Network Modularity', fontsize=16, weight='bold')
    ax.set_xlabel('Network Modularity'); ax.set_ylabel('Average Quarantine Efficiency')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "unified_eff_vs_modularity.svg")); plt.close()
    print("✅ Unified modularity plot saved.")

def plot_snap_containment_metrics(df, output_dir):
    print("\nGenerating SNAP strategy containment metrics plot...")
    snap_nets = [net['name'] for net in config.SNAP_NETWORKS]
    snap_df = df[df['Network_Name'].isin(snap_nets)]
    if snap_df.empty or 'avg_time_to_infection' not in snap_df.columns: return

    plt.figure(figsize=(12, 7))
    sns.barplot(data=snap_df, x='Network_Name', y='avg_time_to_infection', hue='Strategy', palette=STRATEGY_PALETTE, order=snap_nets).set_title('Containment: Avg. Time to First Community Infection', fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.savefig(os.path.join(output_dir, "snap_containment_time.svg")); plt.close()

    plt.figure(figsize=(12, 7))
    sns.barplot(data=snap_df, x='Network_Name', y='proportion_infected', hue='Strategy', palette=STRATEGY_PALETTE, order=snap_nets).set_title('Containment: Proportion of Communities Infected', fontsize=16, weight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.savefig(os.path.join(output_dir, "snap_containment_proportion.svg")); plt.close()
    print("✅ SNAP containment plots saved.")


# --- 5. Main Execution Block ---
if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    summary_df, raw_df = calculate_summary_from_raw_data(RESULTS_DIR)
    
    if summary_df is not None:
        sbm_metrics_df = pd.read_csv(SBM_METRICS_FILE) if os.path.exists(SBM_METRICS_FILE) else calculate_and_save_sbm_metrics(SBM_METRICS_FILE)
        snap_metrics_df = pd.read_csv(SNAP_METRICS_FILE) if os.path.exists(SNAP_METRICS_FILE) else pd.DataFrame()
        network_metrics_df = pd.concat([sbm_metrics_df, snap_metrics_df], ignore_index=True)
        
        rename_map = {'LocalGNTS-14': 'GNTS', 'Beta-Binomial-14': 'MAB-TS'}
        summary_df['Strategy'] = summary_df['Strategy'].replace(rename_map)
        summary_df = summary_df[summary_df['Strategy'] != 'Gamma-Poisson-14']
        summary_with_metrics_df = pd.merge(summary_df, network_metrics_df, on='Network_Name', how='left')
        
        if not raw_df.empty:
            raw_df['Strategy'] = raw_df['Strategy'].replace(rename_map)
            raw_df = raw_df[raw_df['Strategy'] != 'Gamma-Poisson-14']
            plot_performance_curves(raw_df, PLOTS_DIR)

        plot_efficiency_vs_size(summary_with_metrics_df, PLOTS_DIR)
        plot_efficiency_vs_modularity(summary_with_metrics_df, PLOTS_DIR)
        plot_snap_containment_metrics(summary_with_metrics_df, PLOTS_DIR)
        
        print(f"\n✅ Analysis complete.")

