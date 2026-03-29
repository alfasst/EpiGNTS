# network_epidemic.py
# --------------------------------------------------
# Scope:
# - Epidemic dynamics only (SEAIRQ)
# - NO network generation
# - NO dependence on mutable global config for block counts
#
# Option 1 vectorisation:
# - Node state is stored in a NumPy integer array (state_arr) indexed
#   by a contiguous 0..N-1 node index, not as NetworkX node attributes.
# - run_seiqr_step() works entirely on state_arr using vectorised
#   NumPy operations — no Python-level node iteration.
# - The NetworkX graph G is used only for topology (adjacency) and
#   block_id attributes; its 'state' attributes are NOT the live store.
# - A sync helper (sync_state_to_graph) writes state_arr back into G
#   when other code (GNN context builder) needs to read node states.
# - initialize_epidemic() returns (state_arr, node_index) instead of
#   mutating G node attributes.
#
# State encoding (integer):
#   S=0  E=1  I=2  A=3  Q=4  R=5
# --------------------------------------------------

import numpy as np
import networkx as nx
import random
import scipy.sparse as sp

# --------------------------------------------------
# State encoding constants
# --------------------------------------------------
S, E, I, A, Q, R = 0, 1, 2, 3, 4, 5
STATE_NAMES = ['S', 'E', 'I', 'A', 'Q', 'R']
STATE_MAP   = {name: idx for idx, name in enumerate(STATE_NAMES)}


# --------------------------------------------------
# Graph pre-computation
# (call once per graph, reuse across all simulation runs)
# --------------------------------------------------

def build_sim_structures(G: nx.Graph):
    """Pre-compute all static structures needed for vectorised simulation.

    Returns
    -------
    node_index : dict  {node_id -> contiguous int index 0..N-1}
    nodes      : list  ordered node ids (index -> node_id)
    adj_csr    : scipy.sparse.csr_matrix  (N x N) adjacency, float32
    block_arr  : np.ndarray int32 (N,)   block_id per node (-1 if unset)
    """
    nodes      = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    N          = len(nodes)

    # Build CSR adjacency matrix (symmetric, unweighted)
    rows, cols = [], []
    for u, v in G.edges():
        i, j = node_index[u], node_index[v]
        rows += [i, j]
        cols += [j, i]
    data   = np.ones(len(rows), dtype=np.float32)
    adj_csr = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)

    # Block id array
    block_arr = np.full(N, -1, dtype=np.int32)
    for n, d in G.nodes(data=True):
        bid = d.get('block_id')
        if isinstance(bid, int):
            block_arr[node_index[n]] = bid

    return node_index, nodes, adj_csr, block_arr


# --------------------------------------------------
# Epidemic initialisation
# --------------------------------------------------

def initialize_epidemic(N: int, num_initial: int):
    """Create a fresh state array with num_initial nodes set to I.

    Parameters
    ----------
    N           : total number of nodes
    num_initial : number of initially infected nodes

    Returns
    -------
    state_arr : np.ndarray int8 (N,)  — all S except num_initial I nodes
    """
    state_arr = np.full(N, S, dtype=np.int8)
    num_initial = min(num_initial, N)
    if num_initial > 0:
        seeds = np.random.choice(N, size=num_initial, replace=False)
        state_arr[seeds] = I
    return state_arr


# --------------------------------------------------
# State ↔ graph sync
# --------------------------------------------------

def sync_state_to_graph(G: nx.Graph, state_arr: np.ndarray, nodes: list):
    """Write state_arr back into G node attributes.

    Only called before GNN context construction so the GNN can read
    G.nodes[n]['state'] as usual. Not called on every simulation step.
    """
    state_attrs = {nodes[i]: STATE_NAMES[state_arr[i]] for i in range(len(nodes))}
    nx.set_node_attributes(G, state_attrs, 'state')


# --------------------------------------------------
# Vectorised SEAIRQ step
# --------------------------------------------------

def run_seiqr_step(
    state_arr:        np.ndarray,
    adj_csr:          sp.csr_matrix,
    block_arr:        np.ndarray,
    beta:             float,
    sigma:            float,
    gamma:            float,
    asymptomatic_prob: float,
    hub_id:           int,
    hub_multiplier:   float,
    long_range_prob:  float,
    waning_prob:      float,
):
    """Advance epidemic state by one day — fully vectorised.

    Parameters
    ----------
    state_arr  : int8 array (N,), modified in-place
    adj_csr    : CSR adjacency matrix (N x N)
    block_arr  : int32 array (N,) of block ids
    All other parameters match the old signature exactly.

    Epidemic logic is identical to the original node-level loop:
    - Each I/A node transmits to S neighbours with probability eff_beta
    - Hub block nodes use beta * hub_multiplier
    - Long-range spark: each I/A node fires a global random S exposure
      with probability long_range_prob
    - E -> I or A with probability sigma
    - I/Q -> R with probability gamma
    - R -> S with probability waning_prob
    - Newly exposed S nodes -> E (resolved after all transitions)
    """
    N = len(state_arr)

    # Boolean masks for each compartment
    is_S = (state_arr == S)
    is_E = (state_arr == E)
    is_I = (state_arr == I)
    is_A = (state_arr == A)
    is_Q = (state_arr == Q)
    is_R = (state_arr == R)

    infectious = is_I | is_A           # (N,) bool
    inf_idx    = np.where(infectious)[0]
    S_idx      = np.where(is_S)[0]

    new_state  = state_arr.copy()

    # --------------------------------------------------
    # 1. Local transmission: S neighbours of I/A nodes
    # --------------------------------------------------
    # Per-node effective beta (hub block gets multiplier)
    num_blocks   = int(block_arr.max()) + 1 if block_arr.max() >= 0 else 0
    is_hub_valid = 0 <= hub_id < num_blocks
    eff_beta_vec = np.where(
        (block_arr == hub_id) & is_hub_valid,
        beta * hub_multiplier,
        beta,
    ).astype(np.float32)                # (N,)

    if inf_idx.size > 0 and S_idx.size > 0:
        # For each infectious node, get its S neighbours via sparse row slice.
        # Vectorised approach: build a weighted exposure matrix.
        #
        # exposure_pressure[j] = probability that node j (S) is NOT exposed
        # by any single infectious neighbour i = product(1 - eff_beta[i])
        # over all infectious i adjacent to j.
        #
        # Using log: log(1-p) sum over infectious neighbours, then exponentiate.
        # This avoids a Python loop over infectious nodes entirely.

        # Sparse matrix of infectious nodes only: shape (|inf| x N)
        inf_adj = adj_csr[inf_idx]           # (|inf| x N) CSR

        # Weight each infectious node's row by its log(1 - eff_beta)
        log1m_beta = np.log1p(-eff_beta_vec[inf_idx])   # (|inf|,)

        # Multiply each row by its log(1-beta): sparse row scaling
        # Result: (|inf| x N) where entry [i,j] = log(1-beta_i) if edge exists
        scale_diag = sp.diags(log1m_beta, format='csr')
        weighted   = scale_diag.dot(inf_adj)  # (|inf| x N)

        # Sum log(1-beta) contributions over all infectious neighbours per node
        log_no_expose = np.asarray(weighted.sum(axis=0)).ravel()  # (N,)

        # Probability of being exposed by at least one infectious neighbour
        prob_expose = 1.0 - np.exp(log_no_expose)  # (N,)

        # Apply only to S nodes; draw Bernoulli
        expose_draw   = np.random.random(N)
        newly_exposed = is_S & (expose_draw < prob_expose)

        new_state[newly_exposed] = E

    # --------------------------------------------------
    # 2. Long-range (global spark)
    # --------------------------------------------------
    if inf_idx.size > 0 and S_idx.size > 0:
        # Each infectious node independently fires a long-range spark
        fires      = np.random.random(inf_idx.size) < long_range_prob
        n_sparks   = int(fires.sum())
        if n_sparks > 0:
            targets = np.random.choice(S_idx, size=n_sparks, replace=True)
            # Only expose nodes still S (not already newly_exposed above)
            for t in targets:
                if new_state[t] == S:
                    new_state[t] = E

    # --------------------------------------------------
    # 3. E -> I or A
    # --------------------------------------------------
    e_idx = np.where(is_E)[0]
    if e_idx.size > 0:
        progresses  = np.random.random(e_idx.size) < sigma
        prog_idx    = e_idx[progresses]
        if prog_idx.size > 0:
            asym_draw   = np.random.random(prog_idx.size) < asymptomatic_prob
            new_state[prog_idx[asym_draw]]  = A
            new_state[prog_idx[~asym_draw]] = I

    # --------------------------------------------------
    # 4. I -> R
    # --------------------------------------------------
    i_idx = np.where(is_I)[0]
    if i_idx.size > 0:
        recovers = np.random.random(i_idx.size) < gamma
        new_state[i_idx[recovers]] = R
        
    # --------------------------------------------------
    # 4b. A -> R  (was missing — asymptomatic recover at same rate as I)
    # --------------------------------------------------
    a_idx = np.where(is_A)[0]
    if a_idx.size > 0:
        recovers = np.random.random(a_idx.size) < gamma
        new_state[a_idx[recovers]] = R

    # --------------------------------------------------
    # 5. Q -> R
    # --------------------------------------------------
    q_idx = np.where(is_Q)[0]
    if q_idx.size > 0:
        recovers = np.random.random(q_idx.size) < gamma
        new_state[q_idx[recovers]] = R

    # --------------------------------------------------
    # 6. R -> S  (waning immunity)
    # --------------------------------------------------
    r_idx = np.where(is_R)[0]
    if r_idx.size > 0:
        wanes = np.random.random(r_idx.size) < waning_prob
        new_state[r_idx[wanes]] = S

    # Write back in-place
    state_arr[:] = new_state