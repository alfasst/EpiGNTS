# gnts.py
# Core GNTS implementations (LocalGNTS & GlobalGNTS)
# --------------------------------------------------
# This file contains ONLY the learning-based GNTS logic:
# - Graph encoders (GraphSAGE)
# - Prior booster
# - LocalGNTS and GlobalGNTS agents
#
# Wrappers, heuristics, and non-GNTS strategies must live in strategies.py
#
# Design principles:
# - No network generation / no experiment orchestration
# - No reliance on mutable global block counts
# - Block structure inferred from the graph at construction time only
#
# Changes vs previous version:
# - LocalGNTS.update() / GlobalGNTS.update() use history-informed priors
# - GPU-aware: all tensors placed on DEVICE; .cpu() only for numpy ops
# - DEVICE falls back to CPU if GPU compute capability < sm_70
#
# GlobalGNTS optimisations (this version):
# 1. _update_mappings() eliminated — node_map and block_id cached once
#    at __init__ time; topology never changes during a simulation.
# 2. edge_index precomputed once in build_sim_structures() and passed
#    in at construction — no per-step edge list rebuild.
# 3. Python pooling loop replaced by torch_scatter.scatter_mean —
#    a single fused GPU op that averages node embeddings per block.

import os
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_scatter import scatter_mean

import config

# --------------------------------------------------
# Device resolution  (with sm_60 / P100 fallback)
# --------------------------------------------------

def _resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device('cpu')
    major, minor = torch.cuda.get_device_capability(0)
    if major < 7:
        print(
            f"[gnts] WARNING: GPU sm_{major}{minor} < sm_70 minimum. "
            f"Falling back to CPU."
        )
        return torch.device('cpu')
    return torch.device('cuda')

DEVICE = _resolve_device()
print(f"[gnts] Running on device: {DEVICE}")


# --------------------------------------------------
# GNN building blocks
# --------------------------------------------------

class GraphSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        if edge_index.numel() == 0:
            return torch.zeros(
                (x.size(0), config.GNN_OUTPUT_DIM), device=x.device
            )
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


class PriorBooster(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.fc2(F.relu(self.fc1(x))))


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def build_edge_index(
    G: nx.Graph,
    node_map: dict,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """Build a COO edge index tensor for a (sub)graph on `device`.

    Used by LocalGNTS for per-block subgraphs.
    GlobalGNTS uses the precomputed edge_index from build_sim_structures.
    """
    edges = [
        (node_map[u], node_map[v])
        for u, v in G.edges()
        if u in node_map and v in node_map
    ]
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()


def _accumulate_priors(history, num_blocks: int):
    """Accumulate empirical base alpha/beta from the test buffer.

    Returns CPU float32 numpy arrays — these feed np.random.beta and
    torch.tensor() calls that require CPU scalars.
    """
    base_alphas = np.ones(num_blocks, dtype=np.float32)
    base_betas  = np.ones(num_blocks, dtype=np.float32)
    if history:
        for daily in history:
            for i in range(min(num_blocks, len(daily))):
                base_alphas[i] += daily[i]['positive']
                base_betas[i]  += daily[i]['negative']
    return base_alphas, base_betas


# --------------------------------------------------
# Local GNTS
# --------------------------------------------------

class LocalGNTS:
    def __init__(
        self,
        G: nx.Graph,
        num_blocks: int,
        gnn_out_dim: int,
        context_dim: int,
        weight_decay: float,
        device: torch.device = DEVICE,
    ):
        self.G          = G
        self.num_blocks = num_blocks
        self.device     = device

        self.local_gnns = nn.ModuleList([
            GraphSAGE(4, 16, gnn_out_dim) for _ in range(num_blocks)
        ]).to(device)

        self.prior_boosters = nn.ModuleList([
            PriorBooster(context_dim) for _ in range(num_blocks)
        ]).to(device)

        params = chain(
            self.local_gnns.parameters(),
            self.prior_boosters.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            params, lr=0.005, weight_decay=weight_decay
        )

    # ----------------------
    # Context construction
    # ----------------------
    def _get_context(self, G: nx.Graph, day: int, sim_days: int) -> torch.Tensor:
        """(num_blocks, context_dim) tensor on self.device."""
        contexts = []
        for bid in range(self.num_blocks):
            nodes = [n for n, d in G.nodes(data=True) if d.get('block_id') == bid]
            if not nodes:
                contexts.append(
                    torch.zeros(config.LOCAL_AGENT_CONTEXT_DIM, device=self.device)
                )
                continue

            sub      = G.subgraph(nodes)
            node_map = {n: i for i, n in enumerate(sub.nodes())}

            x = torch.zeros((len(nodes), 4), device=self.device)
            for i, n in enumerate(sub.nodes()):
                s   = G.nodes[n].get('state', 'S')
                idx = ['I', 'Q', 'R'].index(s) if s in ['I', 'Q', 'R'] else 3
                x[i, idx] = 1.0

            edge_index = build_edge_index(sub, node_map, device=self.device)
            emb        = self.local_gnns[bid](Data(x=x, edge_index=edge_index))
            block_emb  = (
                emb.mean(dim=0) if emb.numel() > 0
                else torch.zeros(config.GNN_OUTPUT_DIM, device=self.device)
            )

            t = torch.tensor([day / sim_days], dtype=torch.float, device=self.device)
            contexts.append(torch.cat([block_emb, t]))

        return torch.stack(contexts)

    # ----------------------
    # Allocation
    # ----------------------
    def get_allocation_proportions(self, G, history, day: int, sim_days: int):
        self.local_gnns.eval()
        self.prior_boosters.eval()
        with torch.no_grad():
            context = self._get_context(G, day, sim_days)

        alphas, betas = _accumulate_priors(history, self.num_blocks)

        boosts = torch.stack([
            self.prior_boosters[i](context[i]) for i in range(self.num_blocks)
        ])
        alphas = np.maximum(alphas + boosts[:, 0].detach().cpu().numpy(), 1e-6)
        betas  = np.maximum(betas  + boosts[:, 1].detach().cpu().numpy(), 1e-6)

        rates = np.random.beta(alphas, betas)
        return (
            rates / rates.sum() if rates.sum() > 0
            else np.ones(self.num_blocks) / self.num_blocks
        )

    # ----------------------
    # Learning
    # ----------------------
    def update(self, G, day: int, sim_days: int, daily_results, history=None):
        self.local_gnns.train()
        self.prior_boosters.train()
        self.optimizer.zero_grad()

        context                  = self._get_context(G, day, sim_days)
        base_alphas, base_betas  = _accumulate_priors(history, self.num_blocks)

        loss  = torch.tensor(0.0, device=self.device)
        valid = 0

        for i in range(self.num_blocks):
            if i >= len(daily_results) or torch.all(context[i] == 0):
                continue
            boosts = self.prior_boosters[i](context[i])
            alpha  = torch.tensor(
                base_alphas[i], dtype=torch.float, device=self.device
            ) + boosts[0]
            beta   = torch.tensor(
                base_betas[i], dtype=torch.float, device=self.device
            ) + boosts[1]
            s = float(daily_results[i]['positive'])
            f = float(daily_results[i]['negative'])
            nll = -(
                torch.lgamma(alpha + s) + torch.lgamma(beta + f)
                - torch.lgamma(alpha + beta + s + f)
                - (torch.lgamma(alpha) + torch.lgamma(beta)
                   - torch.lgamma(alpha + beta))
            )
            if not torch.isnan(nll):
                loss  += nll
                valid += 1

        if valid > 0:
            (loss / valid).backward()
            self.optimizer.step()
            return (loss / valid).item()
        return 0.0

    # ----------------------
    # Persistence
    # ----------------------
    def save_model(self, path: str):
        torch.save({
            'local_gnns':     self.local_gnns.cpu().state_dict(),
            'prior_boosters': self.prior_boosters.cpu().state_dict(),
        }, path)
        self.local_gnns.to(self.device)
        self.prior_boosters.to(self.device)

    def load_model(self, path: str):
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=self.device)
        self.local_gnns.load_state_dict(ckpt['local_gnns'])
        self.prior_boosters.load_state_dict(ckpt['prior_boosters'])
        self.local_gnns.to(self.device)
        self.prior_boosters.to(self.device)


# --------------------------------------------------
# Global GNTS
# --------------------------------------------------

class GlobalGNTS:
    """Graph-level GNTS agent with three performance optimisations:

    1. node_map and block_id_tensor cached once at __init__ —
       _update_mappings() is never called again.
    2. edge_index_t (precomputed COO tensor from build_sim_structures)
       passed in at construction — no per-step edge list rebuild.
    3. scatter_mean (torch_scatter) replaces the Python pooling loop —
       a single fused GPU op averages node embeddings per block.

    Construction
    ------------
    Pass the five values returned by build_sim_structures() as
    `precomputed`:

        node_index, nodes, adj_csr, block_arr, edge_index_t, block_id_t
            = build_sim_structures(G)

        agent = GlobalGNTS(
            G, num_blocks, gnn_out_dim, context_dim, weight_decay,
            precomputed=(node_index, nodes, adj_csr, block_arr,
                         edge_index_t, block_id_t),
        )

    If `precomputed` is None, the structures are built internally from G
    (slower — use only for backward compatibility / tests).
    """

    def __init__(
        self,
        G: nx.Graph,
        num_blocks: int,
        gnn_out_dim: int,
        context_dim: int,
        weight_decay: float,
        device: torch.device = DEVICE,
        precomputed=None,
    ):
        self.num_blocks = num_blocks
        self.device     = device

        self.global_gnn    = GraphSAGE(4, 16, gnn_out_dim).to(device)
        self.prior_booster = PriorBooster(context_dim).to(device)

        params = chain(
            self.global_gnn.parameters(),
            self.prior_booster.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            params, lr=0.005, weight_decay=weight_decay
        )

        # --------------------------------------------------
        # Cache static graph structures — built once, never rebuilt
        # --------------------------------------------------
        if precomputed is not None:
            node_index, nodes, _adj_csr, _block_arr, edge_index_t, block_id_t = precomputed
        else:
            # Fallback: build internally (slower path)
            from network_epidemic import build_sim_structures
            node_index, nodes, _adj_csr, _block_arr, edge_index_t, block_id_t = \
                build_sim_structures(G, device=device)

        # node_map: node_id -> contiguous index (CPU dict, used for feature matrix)
        self.node_map    = node_index          # {node_id: int}
        self.nodes       = nodes               # [node_id, ...]  index -> node_id
        self.N           = len(nodes)

        # Precomputed COO edge index on device — reused every _get_context call
        self.edge_index_t = edge_index_t.to(device)   # (2, E) LongTensor

        # Block id per node on device — fed directly to scatter_mean
        # Mask out -1 entries (nodes with no block) to block 0 for scatter safety;
        # those nodes are excluded from the block embedding by zeroing their weight.
        raw_bid            = block_id_t.to(device)     # (N,) LongTensor
        self.valid_mask    = raw_bid >= 0              # (N,) BoolTensor
        # scatter_mean requires non-negative index; clamp -1 -> 0 (masked later)
        self.block_id_t    = raw_bid.clamp(min=0)      # (N,) LongTensor

    # ----------------------
    # Context construction  (optimised)
    # ----------------------
    def _get_context(self, G: nx.Graph, day: int, sim_days: int) -> torch.Tensor:
        """(num_blocks, context_dim) tensor on self.device.

        Optimisations vs old version
        ----------------------------
        - No _update_mappings() call — node_map is already cached.
        - No edge_index rebuild — self.edge_index_t is precomputed.
        - No Python pooling loop — scatter_mean does block aggregation.
        """
        # ---- Node feature matrix (still needs G.nodes state after sync) ----
        x = torch.zeros((self.N, 4), device=self.device)
        for n, idx in self.node_map.items():
            s    = G.nodes[n].get('state', 'S')
            feat = ['I', 'Q', 'R'].index(s) if s in ['I', 'Q', 'R'] else 3
            x[idx, feat] = 1.0

        # ---- GNN forward — full graph, precomputed edge index ----
        node_emb = self.global_gnn(
            Data(x=x, edge_index=self.edge_index_t)
        )  # (N, gnn_out_dim) on device

        # ---- Block pooling via scatter_mean (single GPU op) ----
        # Zero out embeddings of nodes with no valid block before scattering
        if not self.valid_mask.all():
            node_emb = node_emb * self.valid_mask.unsqueeze(1).float()

        # scatter_mean: for each block b, average node_emb[i] where block_id_t[i]==b
        # out_size = num_blocks ensures every block gets a row even if empty
        block_emb = scatter_mean(
            node_emb,
            self.block_id_t,
            dim=0,
            out=torch.zeros(
                (self.num_blocks, config.GNN_OUTPUT_DIM), device=self.device
            ),
        )  # (num_blocks, gnn_out_dim) on device

        t = torch.full(
            (self.num_blocks, 1), day / sim_days, device=self.device
        )
        return torch.cat([block_emb, t], dim=1)  # (num_blocks, context_dim)

    # ----------------------
    # Allocation
    # ----------------------
    def get_allocation_proportions(self, G, history, day: int, sim_days: int):
        self.global_gnn.eval()
        self.prior_booster.eval()
        with torch.no_grad():
            context = self._get_context(G, day, sim_days)

        alphas, betas = _accumulate_priors(history, self.num_blocks)

        boosts = self.prior_booster(context)   # (num_blocks, 2)
        alphas = np.maximum(alphas + boosts[:, 0].detach().cpu().numpy(), 1e-6)
        betas  = np.maximum(betas  + boosts[:, 1].detach().cpu().numpy(), 1e-6)

        rates = np.random.beta(alphas, betas)
        return (
            rates / rates.sum() if rates.sum() > 0
            else np.ones(self.num_blocks) / self.num_blocks
        )

    # ----------------------
    # Learning
    # ----------------------
    def update(self, G, day: int, sim_days: int, daily_results, history=None):
        """Update global GNN and PriorBooster via Beta-Binomial NLL."""
        self.global_gnn.train()
        self.prior_booster.train()
        self.optimizer.zero_grad()

        context                  = self._get_context(G, day, sim_days)
        base_alphas, base_betas  = _accumulate_priors(history, self.num_blocks)

        loss  = torch.tensor(0.0, device=self.device)
        valid = 0

        for i in range(self.num_blocks):
            if i >= len(daily_results) or torch.all(context[i] == 0):
                continue
            boosts = self.prior_booster(context[i])
            alpha  = torch.tensor(
                base_alphas[i], dtype=torch.float, device=self.device
            ) + boosts[0]
            beta   = torch.tensor(
                base_betas[i], dtype=torch.float, device=self.device
            ) + boosts[1]
            s = float(daily_results[i]['positive'])
            f = float(daily_results[i]['negative'])
            nll = -(
                torch.lgamma(alpha + s) + torch.lgamma(beta + f)
                - torch.lgamma(alpha + beta + s + f)
                - (torch.lgamma(alpha) + torch.lgamma(beta)
                   - torch.lgamma(alpha + beta))
            )
            if not torch.isnan(nll):
                loss  += nll
                valid += 1

        if valid > 0:
            (loss / valid).backward()
            self.optimizer.step()
            return (loss / valid).item()
        return 0.0

    # ----------------------
    # Persistence
    # ----------------------
    def save_model(self, path: str):
        torch.save({
            'global_gnn':    self.global_gnn.cpu().state_dict(),
            'prior_booster': self.prior_booster.cpu().state_dict(),
        }, path)
        self.global_gnn.to(self.device)
        self.prior_booster.to(self.device)

    def load_model(self, path: str):
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=self.device)
        self.global_gnn.load_state_dict(ckpt['global_gnn'])
        self.prior_booster.load_state_dict(ckpt['prior_booster'])
        self.global_gnn.to(self.device)
        self.prior_booster.to(self.device)