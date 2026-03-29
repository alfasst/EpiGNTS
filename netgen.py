# netgen.py
# Network generation & metric computation (SBM + SNAP)
# --------------------------------------------------
# SNAP optimization:
# - Build node → communities map from community file (NO node×community scan)
# - Community-aware sampling (≤10k nodes)
# - Remove singleton communities

import os
import pickle
import collections
import random
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

import config

# --------------------------------------------------
# Constants
# --------------------------------------------------
MAX_NODES_SNAP = 10_000
TOP_N_COMMUNITIES_SNAP = 20

# --------------------------------------------------
# Paths
# --------------------------------------------------
GPICKLE_DIR = os.path.join("results", "gpickle")
METRICS_FILE = os.path.join("results", "network_metrics.csv")

os.makedirs(GPICKLE_DIR, exist_ok=True)

# --------------------------------------------------
# Metric utilities
# --------------------------------------------------

def compute_basic_metrics(G):
    degrees = np.array([d for _, d in G.degree()])
    betweenness = np.array(list(nx.betweenness_centrality(G).values()))
    closeness = np.array(list(nx.closeness_centrality(G).values()))

    return {
        "Avg_Degree": degrees.mean() if len(degrees) else 0.0,
        "Max_Degree": degrees.max() if len(degrees) else 0.0,
        "Avg_Betweenness": betweenness.mean() if len(betweenness) else 0.0,
        "Avg_Closeness": closeness.mean() if len(closeness) else 0.0,
    }


def compute_stiffness(G, communities):
    vals = []
    for comm in communities:
        if len(comm) <= 1:
            continue
        sub = G.subgraph(comm)
        kin = sub.number_of_edges() * 2
        ktot = sum(dict(G.degree(comm)).values())
        if ktot > 0:
            vals.append(kin / ktot)
    return float(np.mean(vals)) if vals else 0.0


def compute_community_metrics(G):
    """
    Compute modularity, average clustering, and stiffness.

    IMPORTANT (SNAP fix):
    - Communities may NOT form a full partition of G because
      singleton communities are removed by design.
    - Therefore, modularity must be computed on the induced
      subgraph of nodes that actually belong to valid communities.
    """
    block_ids = nx.get_node_attributes(G, "block_id")

    # Build communities from block_id
    comms = collections.defaultdict(set)
    for n, b in block_ids.items():
        if isinstance(b, int) and b >= 0:
            comms[b].add(n)

    # Keep only non-singleton communities
    communities = [c for c in comms.values() if len(c) > 1]

    # ---- Modularity (patched) ----
    if communities:
        nodes_in_comms = set().union(*communities)
        G_sub = G.subgraph(nodes_in_comms)
        modularity = nx.community.modularity(G_sub, communities)
    else:
        modularity = 0.0

    # ---- Clustering (community-averaged) ----
    clustering = nx.clustering(G)
    if communities:
        avg_clust = float(np.mean([
            np.mean([clustering[n] for n in c]) for c in communities
        ]))
    else:
        avg_clust = 0.0

    # ---- Stiffness ----
    stiffness = compute_stiffness(G, communities)

    return modularity, avg_clust, stiffness


# --------------------------------------------------
# SBM generator (unchanged)
# --------------------------------------------------

def generate_sbm(net):
    sizes = net['block_sizes']
    p_in, p_out = net['p_in'], net['p_out']

    prob = [[p_out] * len(sizes) for _ in sizes]
    for i in range(len(sizes)):
        prob[i][i] = p_in

    G = nx.stochastic_block_model(sizes, prob, seed=42)

    node_block = {}
    idx = 0
    for bid, size in enumerate(sizes):
        for _ in range(size):
            node_block[idx] = bid
            idx += 1

    nx.set_node_attributes(G, node_block, 'block_id')
    return G


# --------------------------------------------------
# SNAP utilities (OPTIMIZED)
# --------------------------------------------------

def bfs_sample(G, nodes, target_size):
    if len(nodes) <= target_size:
        return set(nodes)

    start = random.choice(list(nodes))
    visited = {start}
    queue = deque([start])

    while queue and len(visited) < target_size:
        u = queue.popleft()
        for v in G.neighbors(u):
            if v in nodes and v not in visited:
                visited.add(v)
                queue.append(v)
                if len(visited) >= target_size:
                    break
    return visited



def load_snap_network(net):
    # ---- Load full SNAP graph ----
    G_full = nx.read_edgelist(net['path'], nodetype=int)

    # ---- Build node → communities map (KEY OPTIMIZATION) ----
    node_to_comms = defaultdict(list)
    communities = []

    with open(net['communities_path']) as f:
        for cid, line in enumerate(f):
            nodes = set(map(int, line.split()))
            if len(nodes) <= 1:
                continue
            communities.append(nodes)
            for n in nodes:
                node_to_comms[n].append(cid)

    # ---- Assign each node to ONE community ----
    node_block = {}
    for node, comm_ids in node_to_comms.items():
        if len(comm_ids) == 1:
            node_block[node] = comm_ids[0]
        else:
            neigh = set(G_full.neighbors(node))
            scores = [
                (cid, len(neigh & communities[cid])) for cid in comm_ids
            ]
            node_block[node] = max(scores, key=lambda x: x[1])[0]

    # ---- Build non-overlapping communities ----
    comm_dict = defaultdict(set)
    for n, cid in node_block.items():
        comm_dict[cid].add(n)

    # ---- Remove singleton communities ----
    comm_dict = {
        cid: nodes for cid, nodes in comm_dict.items()
        if len(nodes) > 1
    }

    # ---- Select top communities ----
    communities = sorted(comm_dict.values(), key=len, reverse=True)
    communities = communities[:TOP_N_COMMUNITIES_SNAP]

    # ---- Sample nodes to enforce MAX_NODES_SNAP ----
    total_nodes = sum(len(c) for c in communities)
    sampled_nodes = set()

    if total_nodes <= MAX_NODES_SNAP:
        for c in communities:
            sampled_nodes |= c
    else:
        for c in communities:
            quota = max(2, int(len(c) / total_nodes * MAX_NODES_SNAP))
            sampled_nodes |= bfs_sample(G_full, c, quota)

    # ---- Induced subgraph ----
    G = G_full.subgraph(sampled_nodes).copy()

    # ---- Relabel nodes ----
    G = nx.convert_node_labels_to_integers(G, label_attribute='original_id')

    # ---- Assign block_id ----
    original_to_block = {}
    for bid, c in enumerate(communities):
        for n in c:
            original_to_block[n] = bid

    block_id = {}
    for n, d in G.nodes(data=True):
        block_id[n] = original_to_block.get(d['original_id'], -1)

    nx.set_node_attributes(G, block_id, 'block_id')
    return G


# --------------------------------------------------
# Main execution
# --------------------------------------------------

if __name__ == '__main__':
    all_metrics = []

    print("Processing SBM networks...")
    for net in tqdm(config.TEST_NETWORKS):
        name = net['name']
        gpath = os.path.join(GPICKLE_DIR, f"{name}.gpickle")

        if os.path.exists(gpath):
            with open(gpath, 'rb') as f:
                G = pickle.load(f)
        else:
            G = generate_sbm(net)
            with open(gpath, 'wb') as f:
                pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

        basic = compute_basic_metrics(G)
        modularity, clustering, stiffness = compute_community_metrics(G)

        all_metrics.append({
            'Network': name,
            'Type': 'SBM',
            'Nodes': G.number_of_nodes(),
            'Edges': G.number_of_edges(),
            'Modularity': modularity,
            'Clustering': clustering,
            'Stiffness': stiffness,
            **basic
        })

    print("Processing SNAP networks...")
    for net in tqdm(config.SNAP_NETWORKS):
        name = net['name']
        gpath = os.path.join(GPICKLE_DIR, f"{name}.gpickle")

        if os.path.exists(gpath):
            with open(gpath, 'rb') as f:
                G = pickle.load(f)
        else:
            G = load_snap_network(net)
            with open(gpath, 'wb') as f:
                pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)

        basic = compute_basic_metrics(G)
        modularity, clustering, stiffness = compute_community_metrics(G)

        all_metrics.append({
            'Network': name,
            'Type': 'SNAP',
            'Nodes': G.number_of_nodes(),
            'Edges': G.number_of_edges(),
            'Modularity': modularity,
            'Clustering': clustering,
            'Stiffness': stiffness,
            **basic
        })

    df = pd.DataFrame(all_metrics)
    df.to_csv(METRICS_FILE, index=False)
    print(f"\nNetwork generation & metrics saved to {METRICS_FILE}")
