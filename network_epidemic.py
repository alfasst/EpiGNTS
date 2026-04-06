# network_epidemic.py

import networkx as nx
import numpy as np
import random
import config

# --- Integer State Encoding ---
S, E, I, A, Q, R = 0, 1, 2, 3, 4, 5
STATE_NAMES = {S: 'S', E: 'E', I: 'I', A: 'A', Q: 'Q', R: 'R'}
STATE_FROM_NAME = {v: k for k, v in STATE_NAMES.items()}


def build_sim_graph(G_nx):
    """
    Convert a NetworkX graph into fast NumPy-based sim structures.
    Called ONCE per network (not per run).

    Returns a dict (sim_graph) containing:
      - n_nodes        : int
      - adj_lists      : list[np.ndarray]  — neighbour arrays per node (CSR-style)
      - block_ids      : np.ndarray[int]   — block id per node
      - block_nodes    : dict[int -> np.ndarray] — precomputed node lists per block
      - num_blocks     : int
    """
    n = G_nx.number_of_nodes()

    # Neighbour lists — precomputed once, never changes
    adj_lists = [np.array(list(G_nx.neighbors(node)), dtype=np.int32)
                 for node in range(n)]

    # Block ids
    block_ids = np.array([G_nx.nodes[node]['block_id'] for node in range(n)],
                         dtype=np.int32)

    num_blocks = int(block_ids.max()) + 1

    # Precompute block -> node arrays
    block_nodes = {b: np.where(block_ids == b)[0].astype(np.int32)
                   for b in range(num_blocks)}

    return {
        'n_nodes':    n,
        'adj_lists':  adj_lists,
        'block_ids':  block_ids,
        'block_nodes': block_nodes,
        'num_blocks': num_blocks,
    }


def make_initial_states(sim_graph, num_initial):
    """
    Return a fresh state array (all S, then seed infections).
    This replaces deepcopy(G) — only the state array is reset per run.
    """
    n = sim_graph['n_nodes']
    states = np.full(n, S, dtype=np.int8)
    infected_nodes = np.random.choice(n, size=num_initial, replace=False)
    states[infected_nodes] = I
    print(f"🦠 Epidemic seeded with {num_initial} infections.")
    return states


def run_seiqr_step(states, sim_graph, beta, sigma, gamma,
                   asymptomatic_prob, hub_id, hub_multiplier,
                   long_range_prob, waning_prob):
    """
    Advance epidemic one day.
    Operates on integer state array; uses precomputed neighbour lists.
    """
    adj_lists = sim_graph['adj_lists']
    block_ids = sim_graph['block_ids']
    n         = sim_graph['n_nodes']

    new_states = states.copy()

    # Masks for current states
    infectious_mask = (states == I) | (states == A)
    infectious_nodes = np.where(infectious_mask)[0]
    susceptible_nodes = np.where(states == S)[0]

    expose_set = set()

    for node in infectious_nodes:
        b = beta * hub_multiplier if block_ids[node] == hub_id else beta
        neighbours = adj_lists[node]
        if len(neighbours):
            s_mask = states[neighbours] == S
            s_neighbours = neighbours[s_mask]
            if len(s_neighbours):
                hits = np.random.random(len(s_neighbours)) < b
                expose_set.update(s_neighbours[hits].tolist())

        # Long-range spark
        if long_range_prob > 0 and len(susceptible_nodes) and random.random() < long_range_prob:
            expose_set.add(int(np.random.choice(susceptible_nodes)))

        # Recovery
        if random.random() < gamma:
            new_states[node] = R

    # Expose
    for node in expose_set:
        if new_states[node] == S:          # guard: not already transitioned
            new_states[node] = E

    # E -> I or A
    e_nodes = np.where(states == E)[0]
    if len(e_nodes):
        progress = np.random.random(len(e_nodes)) < sigma
        progressing = e_nodes[progress]
        if len(progressing):
            asymp = np.random.random(len(progressing)) < asymptomatic_prob
            new_states[progressing[asymp]]  = A
            new_states[progressing[~asymp]] = I

    # Q -> R
    q_nodes = np.where(states == Q)[0]
    if len(q_nodes):
        recover = np.random.random(len(q_nodes)) < gamma
        new_states[q_nodes[recover]] = R

    # R -> S (waning immunity)
    r_nodes = np.where(states == R)[0]
    if len(r_nodes):
        wane = np.random.random(len(r_nodes)) < waning_prob
        new_states[r_nodes[wane]] = S

    states[:] = new_states


# ---------------------------------------------------------------------------
# Legacy helpers kept for compatibility with create_sbm_network callers
# ---------------------------------------------------------------------------

def create_sbm_network(block_sizes, p_in, p_out):
    """Creates an SBM NetworkX graph (called once, result saved to gpickle)."""
    num_nodes = sum(block_sizes)
    prob_matrix = [[p_out] * len(block_sizes) for _ in range(len(block_sizes))]
    for i in range(len(block_sizes)):
        prob_matrix[i][i] = p_in
    G = nx.stochastic_block_model(block_sizes, prob_matrix, seed=42)
    current_size = 0
    for i, size in enumerate(block_sizes):
        for node_idx in range(size):
            G.nodes[current_size + node_idx]['block_id'] = i
        current_size += size
    print(f"✅ Network created with {num_nodes} nodes.")
    return G