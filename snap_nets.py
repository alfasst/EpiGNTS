# snap_nets.py
# Description: Runs experiments on intelligently sampled subgraphs of SNAP networks.
import os
import torch
from tqdm import tqdm
import collections
import numpy as np
import pandas as pd
import csv
import networkx as nx
from networkx.algorithms.community import modularity as nx_modularity
import sys
import random

import config
from simulation import run_simulation
from strategies import LocalGNTS, average_gnts_bandits

# --- Configuration ---
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
SNAP_METRICS_FILE = os.path.join(RESULTS_DIR, "snap_network_metrics.csv")


# --- NEW: Community Loading and Sampling Functions ---

def load_snap_communities(path):
    """Loads overlapping community data from a SNAP community file."""
    print(f"Loading overlapping communities from {path}...")
    communities = []
    with open(path, 'r') as f:
        for line in f:
            nodes = [int(n) for n in line.strip().split()]
            communities.append(frozenset(nodes))
    return communities

def assign_nodes_to_communities(graph, communities):
    """Converts overlapping communities to a non-overlapping partition using Majority Rule."""
    print("Converting overlapping communities to a non-overlapping partition...")
    node_to_communities = collections.defaultdict(list)
    for i, comm in enumerate(communities):
        for node in comm:
            if node in graph:
                node_to_communities[node].append(i)

    node_partition = {}
    for node, comm_indices in tqdm(node_to_communities.items(), desc="Assigning Nodes"):
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
    
    # Invert partition to get community -> nodes mapping
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
        # Only consider neighbors that are also in the original community
        neighbors_in_comm = [n for n in graph.neighbors(node) if n in community_nodes]
        random.shuffle(neighbors_in_comm)
        
        for neighbor in neighbors_in_comm:
            if neighbor not in sampled_nodes:
                sampled_nodes.add(neighbor)
                queue.append(neighbor)
                if len(sampled_nodes) >= target_size:
                    break
    return list(sampled_nodes)

# --- Metric and Data Export Functions ---

def calculate_and_save_snap_metrics(G, communities, network_name, output_path):
    """Calculates and saves structural metrics for the final sampled SNAP network."""
    modularity_score = nx_modularity(G, communities) if communities else 0.0
    local_clustering = nx.clustering(G)
    ccc = np.mean([np.mean([local_clustering[n] for n in c]) for c in communities if c]) if communities else 0.0
    metrics = {"Network_Name": [network_name], "Modularity": [modularity_score], "Clustering": [ccc], "Size": [G.number_of_nodes()]}
    pd.DataFrame(metrics).to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
    print(f"✅ Structural metrics for {network_name} saved.")

def export_raw_daily_data(results_dict, network_name, network_short_name):
    for strategy_name, histories_list in results_dict.items():
        all_runs_data = [{'Day': d, 'Run': r, 'Efficiency': e} for r, h in enumerate(histories_list) for d, e in enumerate([c.get('Q',0)/sum(c.values()) if sum(c.values())>0 else 0 for day_data in h for c in [collections.Counter(dict(collections.ChainMap(*day_data)))]])]
        pd.DataFrame(all_runs_data).to_csv(os.path.join(RESULTS_DIR, f"{network_short_name}_{strategy_name}.csv"), index=False)
    print(f" > Raw daily data exported for network '{network_name}'.")

def export_run_metrics(all_metrics, network_name, network_short_name):
    for strategy, metrics_list in all_metrics.items():
        pd.DataFrame(metrics_list).drop(columns=['daily_allocations']).to_csv(os.path.join(RESULTS_DIR, f"{network_short_name}_{strategy}_metrics.csv"), index=False)
    print(f" > Run-specific metrics exported for network '{network_name}'.")


if __name__ == '__main__':
    target_network_name = sys.argv[1].lower() if len(sys.argv) > 1 else None
    networks_to_run = config.SNAP_NETWORKS
    if target_network_name:
        networks_to_run = [net for net in config.SNAP_NETWORKS if net['name'].lower() == target_network_name]
        if not networks_to_run: sys.exit(f"ERROR: Target network '{target_network_name}' not found.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not target_network_name and os.path.exists(SNAP_METRICS_FILE): os.remove(SNAP_METRICS_FILE)

    for network_params in networks_to_run:
        network_name, short_name = network_params["name"], network_params["short_name"]
        print(f"\n--- Starting Experiment for Network: {network_name} ---")

        # --- 1. Load, Partition, and Sample Network ---
        if not os.path.exists(network_params['path']) or not os.path.exists(network_params['communities_path']):
            print(f"ERROR: Data files not found for {network_name}. Skipping.")
            continue
        
        full_graph = nx.read_edgelist(network_params['path'], nodetype=int)
        overlapping_comms = load_snap_communities(network_params['communities_path'])
        non_overlapping_comms = assign_nodes_to_communities(full_graph, overlapping_comms)
        
        non_overlapping_comms.sort(key=len, reverse=True)
        top_comms = non_overlapping_comms[:config.TOP_N_COMMUNITIES_SNAP]
        
        total_nodes_in_top_comms = sum(len(c) for c in top_comms)
        print(f"Top {len(top_comms)} communities contain {total_nodes_in_top_comms} nodes.")

        final_sample_nodes = set()
        if total_nodes_in_top_comms <= config.MAX_NODES_SNAP:
            print("Total size is within limit. Using all nodes from top communities.")
            for comm in top_comms: final_sample_nodes.update(comm)
        else:
            print(f"Exceeds limit. Sampling proportionately down to {config.MAX_NODES_SNAP} nodes.")
            proportional_total = 0
            for comm in top_comms:
                proportion = len(comm) / total_nodes_in_top_comms
                target_size = int(proportion * config.MAX_NODES_SNAP)
                proportional_total += target_size
                sampled = bfs_sample_community(full_graph, comm, target_size)
                final_sample_nodes.update(sampled)
            print(f"Sampled a total of {len(final_sample_nodes)} nodes.")

        G_template = nx.relabel_nodes(full_graph.subgraph(final_sample_nodes), {n: i for i, n in enumerate(final_sample_nodes)})
        print(f"Final sampled graph: {G_template.number_of_nodes()} nodes, {G_template.number_of_edges()} edges.")

        # Map original community structure to the new, smaller graph
        final_partition_map = {node: i for i, comm in enumerate(top_comms) for node in comm}
        for node, data in G_template.nodes(data=True):
            original_node = list(final_sample_nodes)[node] # This is fragile, needs a better map
            G_template.nodes[node]['block_id'] = final_partition_map.get(original_node)
        
        final_communities = [frozenset(c) for c in collections.defaultdict(list, {cid: [n for n,d in G_template.nodes(data=True) if d['block_id'] == cid] for cid in range(len(top_comms))}).values()]
        config.NUM_BLOCKS = len(final_communities)
        calculate_and_save_snap_metrics(G_template, final_communities, network_name, SNAP_METRICS_FILE)
        
        # --- 2. Training and Testing ---
        model_path = os.path.join(RESULTS_DIR, f"agent_{short_name}.pth")
        master_agent = LocalGNTS(G_template, config.NUM_BLOCKS, config.LOCAL_GNN_OUTPUT_DIM, config.LOCAL_GNTS_CONTEXT_DIM, config.WEIGHT_DECAY)
        
        if os.path.exists(model_path):
            master_agent.load_model(model_path)
        else:
            trained_agents = [run_simulation('LocalGNTS-14', G_template)[1] for _ in tqdm(range(config.N_TRAINING_RUNS), desc=f"Training on {network_name}")]
            master_agent = average_gnts_bandits(trained_agents, master_agent)
            master_agent.save_model(model_path)
        
        strategies = ['LocalGNTS-14', 'Beta-Binomial-14', 'Gamma-Poisson-14', 'Proportional', 'Uniform', 'Random']
        all_test_results, all_test_metrics = collections.defaultdict(list), collections.defaultdict(list)
        for i in tqdm(range(config.N_TESTING_RUNS), desc=f"Testing on {network_name}"):
            for strat in strategies:
                h, _, m = run_simulation(strat, G_template, pretrained_gnts=master_agent)
                all_test_results[strat].append(h); all_test_metrics[strat].append(m)
        
        export_raw_daily_data(all_test_results, network_name, short_name)
        export_run_metrics(all_test_metrics, network_name, short_name)

    print("\n✅ All specified SNAP network experiments are complete.")

