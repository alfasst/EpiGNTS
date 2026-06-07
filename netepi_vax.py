# netepi_vax.py
# ---------------------------------------------------------------------------
# Vaccine-adapted epidemic module. Extends the SEIQR model from
# network_epidemic.py with a Vaccinated (V) compartment.
#
# Key differences from network_epidemic.py:
#   1. New state V = 6  (vaccinated, immune to exposure)
#   2. V nodes are excluded from the susceptible pool
#   3. V -> S waning transition (configurable via WANING_VACCINE_PROB)
#   4. Node feature encoding updated: S, V, I/A, R buckets
#      (Q is excluded — quarantine is a testing-workflow concept,
#       not used in the vaccine workflow)
#   5. make_initial_states_vax initialises with no Q states
#
# Imports from this file (for use in gnts_vax, simulation_vax, vaccination):
#   States     : S, E, I, A, V, R  (integers 0,1,2,3,5,6 — Q=4 unused here)
#   STATE_NAMES_VAX
#   build_sim_graph          (unchanged, re-exported for convenience)
#   make_initial_states_vax
#   run_seiqr_vax_step
# ---------------------------------------------------------------------------

import networkx as nx
import numpy as np
import random
import config

# ---------------------------------------------------------------------------
# Integer state encoding
# Q (4) is kept in the integer space so that sim_graph structures built with
# network_epidemic.py remain compatible, but it is never assigned in the
# vaccine workflow.
# ---------------------------------------------------------------------------
S, E, I, A, Q, R = 0, 1, 2, 3, 4, 5
V = 6                               # Vaccinated — new state

STATE_NAMES_VAX = {
    S: 'S',
    E: 'E',
    I: 'I',
    A: 'A',
    Q: 'Q',   # retained for integer-space compatibility; unused in vax runs
    R: 'R',
    V: 'V',
}

STATE_FROM_NAME_VAX = {v: k for k, v in STATE_NAMES_VAX.items()}


# ---------------------------------------------------------------------------
# build_sim_graph — identical to network_epidemic.py.
# Re-exported here so simulation_vax.py only needs to import from netepi_vax.
# ---------------------------------------------------------------------------

def build_sim_graph(G_nx):
    """
    Convert a NetworkX graph into fast NumPy-based sim structures.
    Called ONCE per network (not per run).

    Returns a dict (sim_graph) containing:
      - n_nodes     : int
      - adj_lists   : list[np.ndarray]  — neighbour arrays per node
      - block_ids   : np.ndarray[int]   — block id per node
      - block_nodes : dict[int -> np.ndarray] — node lists per block
      - num_blocks  : int
    """
    n = G_nx.number_of_nodes()

    adj_lists = [
        np.array(list(G_nx.neighbors(node)), dtype=np.int32)
        for node in range(n)
    ]

    block_ids = np.array(
        [G_nx.nodes[node]['block_id'] for node in range(n)],
        dtype=np.int32
    )

    num_blocks = int(block_ids.max()) + 1

    block_nodes = {
        b: np.where(block_ids == b)[0].astype(np.int32)
        for b in range(num_blocks)
    }

    return {
        'n_nodes':     n,
        'adj_lists':   adj_lists,
        'block_ids':   block_ids,
        'block_nodes': block_nodes,
        'num_blocks':  num_blocks,
    }


# ---------------------------------------------------------------------------
# Initial state — vaccine workflow never seeds Q states
# ---------------------------------------------------------------------------

def make_initial_states_vax(sim_graph, num_initial):
    """
    Return a fresh state array for a vaccine-workflow run.
    All nodes start as S; a random subset is seeded as I.
    V is initialised to zero (no pre-existing vaccination).
    Q is never assigned.
    """
    n = sim_graph['n_nodes']
    states = np.full(n, S, dtype=np.int8)
    infected_nodes = np.random.choice(n, size=num_initial, replace=False)
    states[infected_nodes] = I
    print(f"🦠 Epidemic seeded with {num_initial} infections (vaccine workflow).")
    return states


# ---------------------------------------------------------------------------
# Node feature encoding for GNN (vaccine workflow)
#
# 4-bucket one-hot encoding:
#   col 0 — S  : susceptible (vaccination target)
#   col 1 — V  : vaccinated  (already protected)
#   col 2 — I/A: actively infectious (symptomatic or asymptomatic)
#   col 3 — R  : naturally recovered / immune
#
# E is folded into col 0 (S bucket) — exposed nodes are not yet infectious
# and are indistinguishable from susceptibles without a diagnostic test.
# Q is not used in the vaccine workflow; if somehow present it maps to col 3
# (treated as removed/immune, same as R).
# ---------------------------------------------------------------------------

def build_node_features_vax(states_subset):
    """
    Given a 1-D int8 array of states for a set of nodes,
    return a (N, 4) float32 torch tensor with vaccine-workflow one-hot encoding.

    Bucket mapping:
        S (0), E (1)    -> col 0  (susceptible / exposed; both are vaccine targets)
        V (6)           -> col 1  (vaccinated)
        I (2), A (3)    -> col 2  (infectious)
        R (5), Q (4)    -> col 3  (recovered / removed)
    """
    import torch
    n = len(states_subset)
    feats = torch.zeros(n, 4, dtype=torch.float32)

    # col 0 — susceptible bucket: S and E
    feats[(states_subset == S) | (states_subset == E), 0] = 1.0

    # col 1 — vaccinated
    feats[states_subset == V, 1] = 1.0

    # col 2 — infectious
    feats[(states_subset == I) | (states_subset == A), 2] = 1.0

    # col 3 — recovered / removed (R and Q)
    feats[(states_subset == R) | (states_subset == Q), 3] = 1.0

    return feats


# ---------------------------------------------------------------------------
# Epidemic step — SEIQR + V compartment
#
# Changes vs run_seiqr_step in network_epidemic.py:
#   - V nodes are excluded from the susceptible pool (cannot be exposed)
#   - V -> S waning transition added (uses config.WANING_VACCINE_PROB)
#   - Q transitions retained for integer-space completeness but Q nodes
#     will never appear in vaccine-workflow state arrays
# ---------------------------------------------------------------------------

def run_seiqr_vax_step(
    states, sim_graph,
    beta, sigma, gamma,
    asymptomatic_prob,
    hub_id, hub_multiplier,
    long_range_prob,
    waning_immunity_prob,
    waning_vaccine_prob,
):
    """
    Advance the vaccine-workflow epidemic one day.

    Compartment transitions:
        S  -> E   : exposure via infectious neighbour (beta) or long-range spark
        E  -> I   : progression with prob sigma, symptomatic
        E  -> A   : progression with prob sigma * asymptomatic_prob
        I  -> R   : recovery with prob gamma
        A  -> R   : recovery with prob gamma
        R  -> S   : waning natural immunity with prob waning_immunity_prob
        V  -> S   : waning vaccine immunity with prob waning_vaccine_prob

    V nodes are fully protected: they are excluded from expose_set.
    """
    adj_lists = sim_graph['adj_lists']
    block_ids = sim_graph['block_ids']
    n         = sim_graph['n_nodes']

    new_states = states.copy()

    # Infectious nodes (symptomatic + asymptomatic)
    infectious_mask  = (states == I) | (states == A)
    infectious_nodes = np.where(infectious_mask)[0]

    # Susceptible pool: S only (V excluded, E already exposed)
    susceptible_nodes = np.where(states == S)[0]

    expose_set = set()

    for node in infectious_nodes:
        # Hub block heterogeneity
        b = beta * hub_multiplier if block_ids[node] == hub_id else beta

        neighbours = adj_lists[node]
        if len(neighbours):
            # Only attempt to expose S neighbours (not V)
            s_mask       = states[neighbours] == S
            s_neighbours = neighbours[s_mask]
            if len(s_neighbours):
                hits = np.random.random(len(s_neighbours)) < b
                expose_set.update(s_neighbours[hits].tolist())

        # Long-range spark — targets S pool only
        if (long_range_prob > 0
                and len(susceptible_nodes)
                and random.random() < long_range_prob):
            expose_set.add(int(np.random.choice(susceptible_nodes)))

        # Recovery: I or A -> R
        if random.random() < gamma:
            new_states[node] = R

    # S -> E
    for node in expose_set:
        if new_states[node] == S:          # guard: not already transitioned
            new_states[node] = E

    # E -> I or A
    e_nodes = np.where(states == E)[0]
    if len(e_nodes):
        progress    = np.random.random(len(e_nodes)) < sigma
        progressing = e_nodes[progress]
        if len(progressing):
            asymp = np.random.random(len(progressing)) < asymptomatic_prob
            new_states[progressing[asymp]]  = A
            new_states[progressing[~asymp]] = I

    # Q -> R  (retained for completeness; Q nodes absent in vax workflow)
    q_nodes = np.where(states == Q)[0]
    if len(q_nodes):
        recover = np.random.random(len(q_nodes)) < gamma
        new_states[q_nodes[recover]] = R

    # R -> S  (waning natural immunity)
    r_nodes = np.where(states == R)[0]
    if len(r_nodes):
        wane = np.random.random(len(r_nodes)) < waning_immunity_prob
        new_states[r_nodes[wane]] = S

    # V -> S  (waning vaccine immunity)
    v_nodes = np.where(states == V)[0]
    if len(v_nodes):
        wane_vax = np.random.random(len(v_nodes)) < waning_vaccine_prob
        new_states[v_nodes[wane_vax]] = S

    states[:] = new_states
