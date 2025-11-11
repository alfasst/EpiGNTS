# analyze.py
# Description: Calculates all network structural metrics (SBM and SNAP)
# and runs the full analysis, generating summaries and plots.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import networkx as nx
from networkx.algorithms.community import modularity as nx_modularity
import numpy as np
import ast # For safely evaluating string-formatted dictionaries
import matplotlib.ticker as mticker
import collections
from tqdm import tqdm
import random

import config

# --- 1. Configuration & Setup ---
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# --- Path for the legacy SBM summary file ---
LEGACY_SUMMARY_FILE = os.path.join(RESULTS_DIR, "sbm_summary_legacy.csv")
# --- NEW: Path for the new SNAP summary file ---
SNAP_SUMMARY_FILE = os.path.join(RESULTS_DIR, "strategy_summary_SNAP.csv")

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
    """Helper function to parse string-formatted dictionary columns."""
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

def calculate_snap_summary(results_dir, short_to_long_name):
    """
    Generates a new summary just for SNAP networks from their raw metrics files.
    """
    print("Calculating summary statistics for SNAP networks...")
    snap_metric_files = []
    snap_short_names = [net['short_name'] for net in config.SNAP_NETWORKS]
    
    # Find all metric files belonging to SNAP networks
    for f in glob.glob(os.path.join(results_dir, "*_metrics.csv")):
        basename = os.path.basename(f).replace('_metrics.csv', '')
        parts = basename.split('_')
        strategy, network_short_name = parts[-1], '_'.join(parts[:-1])
        if network_short_name not in short_to_long_name:
             strategy, network_short_name = '_'.join(parts[-2:]), '_'.join(parts[:-2])
        
        if network_short_name in snap_short_names:
            snap_metric_files.append((f, network_short_name, strategy))

    if not snap_metric_files:
        print("No SNAP network metric files found.")
        return pd.DataFrame()

    # Process all found SNAP metric files
    all_metrics_dfs = []
    for f, network_short_name, strategy in snap_metric_files:
        if network_short_name in short_to_long_name:
            try:
                df = pd.read_csv(f)
                df['Network_Name'] = short_to_long_name[network_short_name]
                df['Strategy'] = strategy
                all_metrics_dfs.append(df)
            except pd.errors.EmptyDataError:
                print(f"Warning: Metric file {f} is empty. Skipping.")
    
    if not all_metrics_dfs:
        print("No SNAP data to process.")
        return pd.DataFrame()

    # Concatenate and calculate metrics
    metrics_df = pd.concat(all_metrics_dfs)
    
    # --- NEW: Calculate Positive Test Rate ---
    metrics_df['Positive_Test_Rate'] = metrics_df['total_positive_tests'] / metrics_df['total_tests_administered']
    
    if 'first_infection_day' in metrics_df.columns:
        metrics = metrics_df['first_infection_day'].apply(process_day_dict)
        metrics.columns = ['communities_infected_count', 'proportion_infected', 'avg_time_to_infection']
        metrics_df = pd.concat([metrics_df, metrics], axis=1).drop(columns=['first_infection_day'])
    if 'first_intervention_day' in metrics_df.columns:
        metrics = metrics_df['first_intervention_day'].apply(process_day_dict)
        metrics.columns = ['communities_intervened_count', 'proportion_intervened', 'avg_time_to_intervention']
        metrics_df = pd.concat([metrics_df, metrics], axis=1).drop(columns=['first_intervention_day'])
    
    # Group by network and strategy to create the summary
    snap_summary = metrics_df.groupby(['Network_Name', 'Strategy']).mean(numeric_only=True).reset_index()
    
    # Rename columns for consistency
    snap_summary.rename(columns={
        'peak_infections': 'Avg_Peak_Infections', 
        'time_to_peak': 'Avg_Time_to_Peak',
        'total_new_infections': 'Avg_Total_Infections'
    }, inplace=True, errors='ignore')

    # Note: Avg_Quarantine_Efficiency is added in load_all_data()
    return snap_summary

def load_all_data(results_dir):
    """
    Loads the legacy SBM summary and calculates a new SNAP summary, then combines them.
    Also loads all raw daily data for performance curve plotting.
    """
    
    sbm_map = {v: k for k, v in config.NETWORK_SHORT_NAMES.items()}
    snap_map = {net['short_name']: net['name'] for net in config.SNAP_NETWORKS}
    short_to_long_name = {**sbm_map, **snap_map}

    all_summaries = []

    # --- Part 1: Load legacy SBM summary file ---
    if os.path.exists(LEGACY_SUMMARY_FILE):
        print(f"Loading legacy SBM summary from '{LEGACY_SUMMARY_FILE}'...")
        legacy_summary_df = pd.read_csv(LEGACY_SUMMARY_FILE)
    else:
        print(f"Warning: Legacy SBM summary file not found at '{LEGACY_SUMMARY_FILE}'.")
        legacy_summary_df = pd.DataFrame() # empty df

    # --- Part 2: Calculate new SNAP summary ---
    snap_summary_df = calculate_snap_summary(results_dir, short_to_long_name)
    
    # --- Part 3: Load all raw daily data (for plots and efficiency) ---
    raw_daily_files = [f for f in glob.glob(os.path.join(results_dir, "*.csv")) if '_metrics.csv' not in f and 'summary' not in f and 'network_metrics' not in f]
    if raw_daily_files:
        all_daily_dfs = []
        for f in raw_daily_files:
            basename, strategy = os.path.basename(f).replace('.csv', '').split('_', 1)
            if basename in short_to_long_name:
                try:
                    df = pd.read_csv(f)
                    df['Network_Name'] = short_to_long_name[basename]
                    df['Strategy'] = strategy
                    all_daily_dfs.append(df)
                except pd.errors.EmptyDataError:
                    print(f"Warning: Raw daily file {f} is empty. Skipping.")
        
        if not all_daily_dfs:
            raw_df = pd.DataFrame()
        else:
            raw_df = pd.concat(all_daily_dfs)
            # Calculate efficiency and merge it into the summary
            efficiency_summary = raw_df.groupby(['Network_Name', 'Strategy'])['Efficiency'].mean().reset_index().rename(columns={'Efficiency': 'Avg_Quarantine_Efficiency'})
            
            # Update legacy summary (in memory)
            if not legacy_summary_df.empty:
                legacy_summary_df.set_index(['Network_Name', 'Strategy'], inplace=True)
                legacy_summary_df.update(efficiency_summary.set_index(['Network_Name', 'Strategy']))
                legacy_summary_df.reset_index(inplace=True)
                all_summaries.append(legacy_summary_df)

            # Update SNAP summary, subset, and save
            if not snap_summary_df.empty:
                # --- FIX: Use merge instead of update to add the new column ---
                snap_summary_df = pd.merge(
                    snap_summary_df, 
                    efficiency_summary, 
                    on=['Network_Name', 'Strategy'], 
                    how='left' # Keep all snap summary rows, add efficiency if available
                )
                
                # --- NEW: Subset and save SNAP summary ---
                snap_summary_cols = [
                    'Network_Name', 'Strategy', 'Avg_Quarantine_Efficiency', 
                    'Avg_Peak_Infections', 'Avg_Time_to_Peak', 'Positive_Test_Rate'
                ]
                # Filter to columns that actually exist in the dataframe
                snap_summary_cols_exist = [col for col in snap_summary_cols if col in snap_summary_df.columns]
                snap_summary_final = snap_summary_df[snap_summary_cols_exist]
                snap_summary_final.to_csv(SNAP_SUMMARY_FILE, index=False)
                print(f"✅ SNAP summary data saved to '{SNAP_SUMMARY_FILE}' with requested columns.")
                
                all_summaries.append(snap_summary_df) # Add the full summary for plotting

    else:
        raw_df = pd.DataFrame() # Return empty dataframe if no raw files
    
    if not all_summaries:
        print("Error: No data found to generate summary.")
        return None, None

    # --- Part 4: Combine summaries in memory for plotting ---
    final_summary = pd.concat(all_summaries, ignore_index=True)
    # Standardize column names from legacy/new summaries
    final_summary.rename(columns={'peak_infections': 'Avg_Peak_Infections', 'time_to_peak': 'Avg_Time_to_Peak', 'total_new_infections': 'Avg_Total_Infections'}, inplace=True, errors='ignore')

    print(f"✅ Loaded SBM summary and calculated SNAP summary. Ready for plotting.")
    return final_summary, raw_df


# --- 3. Network Metric Calculation ---

# --- NEW: Network Fitness Calculation ---
def calculate_network_fitness(G, communities_list):
    """Calculates network fitness (stiffness) as the average block-level fitness."""
    num_blocks = len(communities_list)
    if num_blocks == 0:
        return 0.0

    block_fitnesses = []
    for community_nodes in communities_list:
        if not community_nodes:
            continue
        
        # Ensure we are working with nodes present in G
        nodes_in_comm = set(community_nodes).intersection(G.nodes())
        if not nodes_in_comm:
            continue

        k_in_b = 0
        k_out_b = 0
        
        subgraph = G.subgraph(nodes_in_comm)
        k_in_b = subgraph.number_of_edges() * 2 # Sum of internal degrees
        
        total_degree_in_comm = sum(d for n, d in G.degree(nodes_in_comm))
        
        # total_degree = k_in_b (internal) + k_out_b (external)
        k_out_b = total_degree_in_comm - k_in_b
        
        denominator = k_in_b + k_out_b
        if denominator == 0:
            block_fitness = 0.0 # Isolated block with no edges
        else:
            block_fitness = k_in_b / denominator
        
        block_fitnesses.append(block_fitness)

    return np.mean(block_fitnesses) if block_fitnesses else 0.0

def calculate_and_save_sbm_metrics(output_path):
    """Calculates and saves structural metrics for all SBM networks."""
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
        
        # --- NEW: Calculate network fitness ---
        fitness_score = calculate_network_fitness(G, communities)
        
        metrics.append({
            "Network_Name": name, 
            "Modularity": modularity_score, 
            "Clustering": ccc, 
            "Network_Fitness": fitness_score, # Added
            "Size": sum(sizes)
        })
    pd.DataFrame(metrics).to_csv(output_path, index=False)
    print(f"✅ SBM structural metrics (Mod, CCC, Fitness) saved.")
    return pd.DataFrame(metrics)

# --- NEW: Functions to build SNAP graphs and calculate metrics ---
# (These are copied from the original snap_nets.py)

def load_snap_communities(path):
    """Loads overlapping community data from a SNAP community file."""
    # print(f"Loading overlapping communities from {path}...")
    communities = []
    with open(path, 'r') as f:
        for line in f:
            nodes = [int(n) for n in line.strip().split()]
            communities.append(frozenset(nodes))
    return communities

def assign_nodes_to_communities(graph, communities):
    """Converts overlapping communities to a non-overlapping partition using Majority Rule."""
    # print("Converting overlapping communities to a non-overlapping partition...")
    node_to_communities = collections.defaultdict(list)
    for i, comm in enumerate(communities):
        for node in comm:
            if node in graph:
                node_to_communities[node].append(i)

    node_partition = {}
    for node, comm_indices in node_to_communities.items():
        if len(comm_indices) == 1:
            node_partition[node] = comm_indices[0]
        else:
            neighbors = set(graph.neighbors(node))
            max_neighbors = -1
            best_comm = -1
            for comm_idx in comm_indices:
                community_nodes = communities[comm_idx]
                num_neighbors_in_comm = len(neighbors.intersection(community_nodes))
                if num_neighbors_in_comm > max_neighbors:
                    max_neighbors = num_neighbors_in_comm
                    best_comm = comm_idx
            node_partition[node] = best_comm if best_comm != -1 else comm_indices[0]
    
    partition = collections.defaultdict(list)
    for node, comm_id in node_partition.items():
        partition[comm_id].append(node)
    
    return [frozenset(nodes) for nodes in partition.values()]

def bfs_sample_community(graph, community_nodes, target_size):
    """Samples a connected subgraph from a community using BFS."""
    if len(community_nodes) <= target_size:
        return list(community_nodes)
    start_node = random.choice(list(community_nodes))
    queue = collections.deque([start_node])
    sampled_nodes = {start_node}
    while queue and len(sampled_nodes) < target_size:
        node = queue.popleft()
        neighbors_in_comm = [n for n in graph.neighbors(node) if n in community_nodes]
        random.shuffle(neighbors_in_comm)
        for neighbor in neighbors_in_comm:
            if neighbor not in sampled_nodes:
                sampled_nodes.add(neighbor)
                queue.append(neighbor)
                if len(sampled_nodes) >= target_size:
                    break
    return list(sampled_nodes)

def calculate_and_save_snap_metrics(output_path):
    """Loads, builds, and calculates metrics for all SNAP networks."""
    print("Calculating structural metrics for all SNAP networks...")
    metrics = []
    
    for network_params in config.SNAP_NETWORKS:
        network_name, short_name = network_params["name"], network_params["short_name"]
        print(f"  Processing {network_name}...")
        
        if not os.path.exists(network_params['path']) or not os.path.exists(network_params['communities_path']):
            print(f"  ERROR: Data files not found for {network_name}. Skipping.")
            continue
        
        full_graph = nx.read_edgelist(network_params['path'], nodetype=int)
        overlapping_comms = load_snap_communities(network_params['communities_path'])
        non_overlapping_comms = assign_nodes_to_communities(full_graph, overlapping_comms)
        
        non_overlapping_comms.sort(key=len, reverse=True)
        top_comms = non_overlapping_comms[:config.TOP_N_COMMUNITIES_SNAP]
        total_nodes_in_top_comms = sum(len(c) for c in top_comms)

        final_sample_nodes = set()
        if total_nodes_in_top_comms <= config.MAX_NODES_SNAP:
            for comm in top_comms: final_sample_nodes.update(comm)
        else:
            for comm in top_comms:
                proportion = len(comm) / total_nodes_in_top_comms
                target_size = int(proportion * config.MAX_NODES_SNAP)
                sampled = bfs_sample_community(full_graph, comm, target_size)
                final_sample_nodes.update(sampled)

        node_list = sorted(list(final_sample_nodes))
        relabel_map = {old_id: new_id for new_id, old_id in enumerate(node_list)}
        G = nx.relabel_nodes(full_graph.subgraph(node_list), relabel_map)

        final_partition_map = {}
        for i, comm in enumerate(top_comms):
            for node in comm:
                if node in final_sample_nodes:
                    final_partition_map[node] = i

        for new_id, data in G.nodes(data=True):
            original_node = node_list[new_id]
            G.nodes[new_id]['block_id'] = final_partition_map.get(original_node, -1)
        
        communities = [frozenset(c) for c in collections.defaultdict(list, {cid: [n for n,d in G.nodes(data=True) if d['block_id'] == cid] for cid in range(len(top_comms))}).values()]
        
        # Calculate metrics
        modularity_score = nx_modularity(G, communities) if communities else 0.0
        local_clustering = nx.clustering(G)
        ccc = np.mean([np.mean([local_clustering[n] for n in c]) for c in communities if c]) if communities else 0.0
        fitness_score = calculate_network_fitness(G, communities)
        
        metrics.append({
            "Network_Name": network_name, 
            "Modularity": modularity_score, 
            "Clustering": ccc, 
            "Network_Fitness": fitness_score,
            "Size": G.number_of_nodes()
        })
    
    pd.DataFrame(metrics).to_csv(output_path, index=False)
    print(f"✅ SNAP structural metrics (Mod, CCC, Fitness) saved.")
    return pd.DataFrame(metrics)

# --- 4. Plotting Functions ---

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

# --- NEW: Plotting function for Network Fitness ---
def plot_efficiency_vs_fitness(df, output_dir):
    print("\nGenerating unified efficiency vs. network fitness (stiffness) plot...")
    plot_df = df.copy()
    if plot_df.empty or 'Network_Fitness' not in plot_df.columns: 
        print("Skipping: No network fitness data available.")
        return
    
    # NEW: Drop any rows that couldn't have fitness calculated
    plot_df.dropna(subset=['Network_Fitness'], inplace=True)
    if plot_df.empty:
        print("Skipping: No network fitness data available after dropping NaNs.")
        return
    
    plot_df = plot_df.sort_values('Network_Fitness')
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=plot_df, x='Network_Fitness', y='Avg_Quarantine_Efficiency', hue='Strategy', palette=STRATEGY_PALETTE, marker='o')
    ax.set_title('Strategy Efficiency vs. Network Fitness (Stiffness)', fontsize=16, weight='bold')
    ax.set_xlabel('Network Fitness (Stiffness)'); ax.set_ylabel('Average Quarantine Efficiency')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "unified_eff_vs_fitness.svg")); plt.close()
    print("✅ Unified network fitness plot saved.")


# --- 5. Main Execution Block ---
if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --- NEW: Calculate and save all network metrics first ---
    sbm_metrics_df = calculate_and_save_sbm_metrics(SBM_METRICS_FILE)
    snap_metrics_df = calculate_and_save_snap_metrics(SNAP_METRICS_FILE)
    network_metrics_df = pd.concat([sbm_metrics_df, snap_metrics_df], ignore_index=True)

    # --- Load simulation results and create summaries ---
    summary_df, raw_df = load_all_data(RESULTS_DIR)
    
    if summary_df is not None:
        # Clean up strategy names for plotting
        rename_map = {'LocalGNTS-14': 'GNTS', 'Beta-Binomial-14': 'MAB-TS'}
        summary_df['Strategy'] = summary_df['Strategy'].replace(rename_map)
        summary_df = summary_df[summary_df['Strategy'] != 'Gamma-Poisson-14']
        
        # Merge the summary data with the structural metrics
        summary_with_metrics_df = pd.merge(summary_df, network_metrics_df, on='Network_Name', how='left')
        
        # Process raw daily data for plotting (if it exists)
        if not raw_df.empty:
            raw_df['Strategy'] = raw_df['Strategy'].replace(rename_map)
            raw_df = raw_df[raw_df['Strategy'] != 'Gamma-Poisson-14']
            # plot_performance_curves(raw_df, PLOTS_DIR) # Removed as requested

        # Generate all plots
        plot_efficiency_vs_size(summary_with_metrics_df, PLOTS_DIR)
        plot_efficiency_vs_modularity(summary_with_metrics_df, PLOTS_DIR)
        plot_efficiency_vs_fitness(summary_with_metrics_df, PLOTS_DIR) # Added
        # plot_snap_containment_metrics(summary_with_metrics_df, PLOTS_DIR) # Removed as requested
        
        print(f"\n✅ Analysis complete.")
