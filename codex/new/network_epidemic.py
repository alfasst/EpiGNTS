# network_epidemic.py
# --------------------------------------------------
# Scope:
# - Epidemic dynamics only (SEAIRQ)
# - NO network generation
# - NO dependence on mutable global config for block counts
#
# Vectorisation (Option 1):
# - Node state lives in a NumPy int8 array (state_arr), not in G attributes.
# - run_seiqr_step() is fully vectorised via NumPy / scipy sparse ops.
# - sync_state_to_graph() writes state_arr back into G only when the GNN
#   context builder needs to read node states.
#
# GlobalGNTS optimisation additions:
# - build_sim_structures() now also returns:
#     edge_index_t  : torch.LongTensor (2, E) on DEVICE — precomputed COO
#                     edge index for the full graph; cached once by GlobalGNTS
#     block_id_t    : torch.LongTensor (N,) on DEVICE — block id per node;
#                     used by scatter_mean to pool node embeddings per block
# - Both tensors are static (topology never changes) and live on GPU
#   for the lifetime of the experiment.
#
# State encoding:   S=0  E=1  I=2  A=3  Q=4  R=5
#
# NO CHANGES required from the bug-fix pass (gnts.py / simulation.py).
# The epidemic logic is faithful to the original node-level loop:
# - All six transitions preserved: S->E, E->I/A, I->R, A->R, Q->R, R->S
# - Hub-beta heterogeneity preserved
# - Vectorised sparse matmul is numerically equivalent to the per-node loop
#   (computes the same independent-exposure probability via log-sum trick)
# --------------------------------------------------

import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch

# --------------------------------------------------
# Device  (mirrors gnts.py — resolved once at import)
# --------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def build_sim_structures(G: nx.Graph, device: torch.device = DEVICE):
    """Pre-compute all static structures needed for simulation and GNN.

    Parameters
    ----------
    G      : nx.Graph with 'block_id' node attributes
    device : torch device for tensor outputs

    Returns
    -------
    node_index   : dict   {node_id -> int index 0..N-1}
    nodes        : list   ordered node ids  (index -> node_id)
    adj_csr      : scipy.sparse.csr_matrix (N x N) float32  — epidemic step
    block_arr    : np.ndarray int32 (N,)   block_id per node (-1 if unset)
    edge_index_t : torch.LongTensor (2, E) on `device`  — GNN edge index
    block_id_t   : torch.LongTensor (N,)  on `device`   — block id per node
                   (-1 entries are masked out by GlobalGNTS before scatter)
    """
    nodes      = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}
    N          = len(nodes)

    # ---- scipy CSR adjacency for epidemic step ----
    rows, cols = [], []
    for u, v in G.edges():
        i, j = node_index[u], node_index[v]
        rows += [i, j]
        cols += [j, i]
    data    = np.ones(len(rows), dtype=np.float32)
    adj_csr = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)

    # ---- numpy block array for epidemic step ----
    block_arr = np.full(N, -1, dtype=np.int32)
    for n, d in G.nodes(data=True):
        bid = d.get('block_id')
        if isinstance(bid, int):
            block_arr[node_index[n]] = bid

    # ---- torch COO edge index for GNN (symmetric, same edges as CSR) ----
    if rows:
        edge_index_t = torch.tensor(
            [rows, cols], dtype=torch.long, device=device
        )
    else:
        edge_index_t = torch.empty((2, 0), dtype=torch.long, device=device)

    # ---- torch block id tensor for scatter_mean pooling ----
    block_id_t = torch.tensor(block_arr, dtype=torch.long, device=device)

    return node_index, nodes, adj_csr, block_arr, edge_index_t, block_id_t


# --------------------------------------------------
# Epidemic initialisation
# --------------------------------------------------

def initialize_epidemic(N: int, num_initial: int) -> np.ndarray:
    """Create a fresh state array with num_initial nodes set to I (=2).

    Returns
    -------
    state_arr : np.ndarray int8 (N,)
    """
    state_arr   = np.full(N, S, dtype=np.int8)
    num_initial = min(num_initial, N)
    if num_initial > 0:
        seeds = np.random.choice(N, size=num_initial, replace=False)
        state_arr[seeds] = I
    return state_arr


# --------------------------------------------------
# State ↔ graph sync
# --------------------------------------------------

def sync_state_to_graph(G: nx.Graph, state_arr: np.ndarray, nodes: list):
    """Write state_arr back into G node 'state' attributes (string form).

    Called only immediately before GNN context construction — not every step.
    """
    nx.set_node_attributes(
        G,
        {nodes[i]: STATE_NAMES[int(state_arr[i])] for i in range(len(nodes))},
        'state',
    )


# --------------------------------------------------
# Vectorised SEAIRQ step
# --------------------------------------------------

def run_seiqr_step(
    state_arr:         np.ndarray,
    adj_csr:           sp.csr_matrix,
    block_arr:         np.ndarray,
    beta:              float,
    sigma:             float,
    gamma:             float,
    asymptomatic_prob: float,
    hub_id:            int,
    hub_multiplier:    float,
    long_range_prob:   float,
    waning_prob:       float,
):
    """Advance epidemic state by one day — fully vectorised (in-place).

    Epidemic logic is identical to the original node-level loop.
    """
    N = len(state_arr)

    is_S = (state_arr == S)
    is_E = (state_arr == E)
    is_I = (state_arr == I)
    is_A = (state_arr == A)
    is_Q = (state_arr == Q)
    is_R = (state_arr == R)

    infectious = is_I | is_A
    inf_idx    = np.where(infectious)[0]
    S_idx      = np.where(is_S)[0]

    new_state = state_arr.copy()

    # --------------------------------------------------
    # 1. Local transmission — vectorised via sparse matmul
    #
    # Equivalent to the old per-node loop:
    #   for each infectious node u, for each susceptible neighbour v:
    #       if random() < beta: expose v
    # The vectorised form computes the probability that v is NOT exposed
    # by ANY infectious neighbour (product of independent survival probs),
    # then draws a single Bernoulli per susceptible node.
    # --------------------------------------------------
    num_blocks   = int(block_arr.max()) + 1 if block_arr.max() >= 0 else 0
    is_hub_valid = 0 <= hub_id < num_blocks
    eff_beta_vec = np.where(
        (block_arr == hub_id) & is_hub_valid,
        beta * hub_multiplier,
        beta,
    ).astype(np.float32)

    if inf_idx.size > 0 and S_idx.size > 0:
        inf_adj    = adj_csr[inf_idx]
        log1m_beta = np.log1p(-eff_beta_vec[inf_idx])
        scale_diag = sp.diags(log1m_beta, format='csr')
        weighted   = scale_diag.dot(inf_adj)

        log_no_expose = np.asarray(weighted.sum(axis=0)).ravel()
        prob_expose   = 1.0 - np.exp(log_no_expose)

        expose_draw   = np.random.random(N)
        newly_exposed = is_S & (expose_draw < prob_expose)
        new_state[newly_exposed] = E

    # --------------------------------------------------
    # 2. Long-range spark
    # --------------------------------------------------
    if inf_idx.size > 0 and S_idx.size > 0:
        fires    = np.random.random(inf_idx.size) < long_range_prob
        n_sparks = int(fires.sum())
        if n_sparks > 0:
            targets = np.random.choice(S_idx, size=n_sparks, replace=True)
            for t in targets:
                if new_state[t] == S:
                    new_state[t] = E

    # --------------------------------------------------
    # 3. E -> I or A
    # --------------------------------------------------
    e_idx = np.where(is_E)[0]
    if e_idx.size > 0:
        progresses = np.random.random(e_idx.size) < sigma
        prog_idx   = e_idx[progresses]
        if prog_idx.size > 0:
            asym_draw = np.random.random(prog_idx.size) < asymptomatic_prob
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
    # 4b. A -> R  (asymptomatic recover at same rate as I)
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

    state_arr[:] = new_state