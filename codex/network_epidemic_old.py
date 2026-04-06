# network_epidemic.py

import networkx as nx
import random
import config

def create_sbm_network(block_sizes, p_in, p_out):
    """CHANGED: Creates an SBM network from a list of block sizes."""
    num_nodes = sum(block_sizes)
    prob_matrix = [[p_out] * len(block_sizes) for _ in range(len(block_sizes))]
    for i in range(len(block_sizes)):
        prob_matrix[i][i] = p_in
    
    G = nx.stochastic_block_model(block_sizes, prob_matrix, seed=42)
    
    current_size = 0
    for i, size in enumerate(block_sizes):
        for node_idx in range(size):
            node_id = current_size + node_idx
            G.nodes[node_id]['block_id'] = i
        current_size += size
            
    print(f"✅ Network created with {num_nodes} nodes and uneven blocks.")
    return G

def initialize_epidemic(G, num_initial):
    """Initializes node states for the SEAIRQ model."""
    for node in G.nodes():
        G.nodes[node]['state'] = 'S'
    initial_nodes = random.sample(list(G.nodes()), num_initial)
    for node in initial_nodes:
        G.nodes[node]['state'] = 'I'
    print(f"🦠 Epidemic seeded with {num_initial} infections.")

def run_seiqr_step(G, beta, sigma, gamma, asymptomatic_prob, hub_id, hub_multiplier, long_range_prob, waning_prob):
    """CHANGED: Advances the epidemic, now including R -> S waning immunity."""
    new_states = {}
    nodes_to_become_exposed = set()
    all_susceptible = [n for n, d in G.nodes(data=True) if d['state'] == 'S']

    for node, data in G.nodes(data=True):
        state = data['state']
        if state in ['I', 'A']:
            current_beta = beta * hub_multiplier if data['block_id'] == hub_id else beta
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['state'] == 'S' and random.random() < current_beta:
                    nodes_to_become_exposed.add(neighbor)
            if random.random() < long_range_prob and all_susceptible:
                spark_victim = random.choice(all_susceptible)
                nodes_to_become_exposed.add(spark_victim)
            if random.random() < gamma:
                new_states[node] = 'R'

        elif state == 'E':
            if random.random() < sigma:
                if random.random() < asymptomatic_prob:
                    new_states[node] = 'A'
                else:
                    new_states[node] = 'I'
        
        elif state == 'Q':
            if random.random() < gamma:
                new_states[node] = 'R'
        
        # NEW: Recovered individuals can become susceptible again
        elif state == 'R':
            if random.random() < waning_prob:
                new_states[node] = 'S'

    for node in nodes_to_become_exposed:
        if node not in new_states:
            new_states[node] = 'E'
            
    for node, state in new_states.items():
        G.nodes[node]['state'] = state

