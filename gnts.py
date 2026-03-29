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
# - No network generation
# - No experiment orchestration
# - No reliance on mutable global block counts
# - Block structure inferred from the graph
#
# Changes vs previous version:
# - LocalGNTS.update() and GlobalGNTS.update() train against an informed
#   prior accumulated from the full test buffer (history) rather than 1.0
#
# GPU changes for Kaggle:
# - DEVICE = torch.device('cuda' if available, else 'cpu') resolved once
#   at module load time and shared by all agents
# - All nn.Module instances moved to DEVICE at init via .to(DEVICE)
# - Every tensor created inside _get_context(), update(), and
#   get_allocation_proportions() is explicitly placed on DEVICE
# - build_edge_index() accepts an optional device argument
# - All .detach().cpu().numpy() calls are preserved so numpy/scipy
#   operations (Beta sampling, prior accumulation) stay on CPU
# - save_model() moves modules to CPU before serialising so checkpoints
#   are portable regardless of where they were created, then restores
#   them to DEVICE
# - load_model() maps directly to DEVICE so the loaded model is
#   immediately ready for inference on the current hardware

import os
from collections import defaultdict
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data

import config

# --------------------------------------------------
# Device resolution (done once at import time)
# --------------------------------------------------

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[gnts] Running on device: {DEVICE}")


# --------------------------------------------------
# GNN building blocks
# --------------------------------------------------

class GraphSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        if edge_index.numel() == 0:
            # Return a zero embedding on the correct device
            return torch.zeros((x.size(0), config.GNN_OUTPUT_DIM), device=x.device)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class PriorBooster(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        return F.softplus(self.fc2(x))


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def build_edge_index(
    G: nx.Graph,
    node_map: dict,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """Build a COO edge index tensor placed on `device`."""
    edges = [
        (node_map[u], node_map[v])
        for u, v in G.edges()
        if u in node_map and v in node_map
    ]
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()


def _accumulate_priors(history, num_blocks: int):
    """Accumulate base alpha/beta priors from the full test buffer.

    Returns two float32 numpy arrays of shape (num_blocks,) initialised
    at 1.0 (uniform Beta prior) and incremented by positive/negative
    counts from every day in history.

    Kept on CPU / numpy intentionally — these values feed np.random.beta
    and torch.tensor() wrappers, both of which expect CPU scalars.
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

        # Build ModuleLists and move to device immediately
        self.local_gnns = nn.ModuleList([
            GraphSAGE(4, 16, gnn_out_dim) for _ in range(num_blocks)
        ]).to(device)

        self.prior_boosters = nn.ModuleList([
            PriorBooster(context_dim) for _ in range(num_blocks)
        ]).to(device)

        params = chain(self.local_gnns.parameters(), self.prior_boosters.parameters())
        self.optimizer = torch.optim.Adam(params, lr=0.005, weight_decay=weight_decay)

    # ----------------------
    # Context construction
    # ----------------------
    def _get_context(self, G: nx.Graph, day: int, sim_days: int) -> torch.Tensor:
        """Build a (num_blocks, context_dim) tensor on self.device."""
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

            # Node feature matrix built directly on device
            x = torch.zeros((len(nodes), 4), device=self.device)
            for i, n in enumerate(sub.nodes()):
                s   = G.nodes[n].get('state', 'S')
                idx = ['I', 'Q', 'R'].index(s) if s in ['I', 'Q', 'R'] else 3
                x[i, idx] = 1.0

            edge_index = build_edge_index(sub, node_map, device=self.device)
            data       = Data(x=x, edge_index=edge_index)

            emb       = self.local_gnns[bid](data)
            block_emb = (
                emb.mean(dim=0)
                if emb.numel() > 0
                else torch.zeros(config.GNN_OUTPUT_DIM, device=self.device)
            )

            t = torch.tensor([day / sim_days], dtype=torch.float, device=self.device)
            contexts.append(torch.cat([block_emb, t]))

        return torch.stack(contexts)  # (num_blocks, context_dim) on self.device

    # ----------------------
    # Allocation
    # ----------------------
    def get_allocation_proportions(self, G, history, day: int, sim_days: int):
        self.local_gnns.eval()
        self.prior_boosters.eval()
        with torch.no_grad():
            context = self._get_context(G, day, sim_days)

        alphas, betas = _accumulate_priors(history, self.num_blocks)

        # Boosts computed on device, pulled to CPU for numpy Beta sampling
        boosts = torch.stack([
            self.prior_boosters[i](context[i]) for i in range(self.num_blocks)
        ])
        alphas = np.maximum(alphas + boosts[:, 0].detach().cpu().numpy(), 1e-6)
        betas  = np.maximum(betas  + boosts[:, 1].detach().cpu().numpy(), 1e-6)

        rates = np.random.beta(alphas, betas)
        return (
            rates / rates.sum()
            if rates.sum() > 0
            else np.ones(self.num_blocks) / self.num_blocks
        )

    # ----------------------
    # Learning
    # ----------------------
    def update(self, G, day: int, sim_days: int, daily_results, history=None):
        """Update GNNs and PriorBoosters via Beta-Binomial NLL.

        base_alphas/base_betas are accumulated from the full test buffer
        (history) before the GNN boost is added, so the gradient signal
        reflects the empirical posterior built up over all previous days.
        """
        self.local_gnns.train()
        self.prior_boosters.train()
        self.optimizer.zero_grad()

        context = self._get_context(G, day, sim_days)

        base_alphas, base_betas = _accumulate_priors(history, self.num_blocks)

        # Loss scalar lives on device so .backward() propagates correctly
        loss  = torch.tensor(0.0, device=self.device)
        valid = 0

        for i in range(self.num_blocks):
            if i >= len(daily_results) or torch.all(context[i] == 0):
                continue

            boosts = self.prior_boosters[i](context[i])

            # Wrap CPU numpy scalars as device tensors before arithmetic
            alpha = (
                torch.tensor(base_alphas[i], dtype=torch.float, device=self.device)
                + boosts[0]
            )
            beta = (
                torch.tensor(base_betas[i], dtype=torch.float, device=self.device)
                + boosts[1]
            )

            s = float(daily_results[i]['positive'])
            f = float(daily_results[i]['negative'])

            nll = -(
                torch.lgamma(alpha + s) + torch.lgamma(beta + f)
                - torch.lgamma(alpha + beta + s + f)
                - (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta))
            )

            if not torch.isnan(nll):
                loss  += nll
                valid += 1

        if valid > 0:
            loss = loss / valid
            loss.backward()
            self.optimizer.step()
            return loss.item()
        return 0.0

    # ----------------------
    # Persistence
    # ----------------------
    def save_model(self, path: str):
        # Move to CPU before serialising — checkpoints must be portable
        torch.save(
            {
                'local_gnns':     self.local_gnns.cpu().state_dict(),
                'prior_boosters': self.prior_boosters.cpu().state_dict(),
            },
            path,
        )
        # Restore to original device after saving
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

        # Build modules and move to device immediately
        self.global_gnn    = GraphSAGE(4, 16, gnn_out_dim).to(device)
        self.prior_booster = PriorBooster(context_dim).to(device)

        params = chain(self.global_gnn.parameters(), self.prior_booster.parameters())
        self.optimizer = torch.optim.Adam(params, lr=0.005, weight_decay=weight_decay)

        self._update_mappings(G)

    def _update_mappings(self, G: nx.Graph):
        self.node_map      = {n: i for i, n in enumerate(G.nodes())}
        self.node_to_block = nx.get_node_attributes(G, 'block_id')

    # ----------------------
    # Context construction
    # ----------------------
    def _get_context(self, G: nx.Graph, day: int, sim_days: int) -> torch.Tensor:
        """Build a (num_blocks, context_dim) tensor on self.device."""
        self._update_mappings(G)
        N = G.number_of_nodes()

        # Full-graph node feature matrix built directly on device
        x = torch.zeros((N, 4), device=self.device)
        for n, i in self.node_map.items():
            s   = G.nodes[n].get('state', 'S')
            idx = ['I', 'Q', 'R'].index(s) if s in ['I', 'Q', 'R'] else 3
            x[i, idx] = 1.0

        edge_index = build_edge_index(G, self.node_map, device=self.device)
        data       = Data(x=x, edge_index=edge_index)
        node_emb   = self.global_gnn(data)  # (N, gnn_out_dim) on device

        # Pool node embeddings into per-block embeddings (stays on device)
        block_emb = torch.zeros(
            (self.num_blocks, config.GNN_OUTPUT_DIM), device=self.device
        )
        counts = defaultdict(int)
        for n, i in self.node_map.items():
            bid = self.node_to_block.get(n)
            if isinstance(bid, int) and 0 <= bid < self.num_blocks:
                block_emb[bid] += node_emb[i]
                counts[bid]    += 1

        for bid, c in counts.items():
            block_emb[bid] /= max(c, 1)

        t = torch.full((self.num_blocks, 1), day / sim_days, device=self.device)
        return torch.cat([block_emb, t], dim=1)  # (num_blocks, context_dim) on device

    # ----------------------
    # Allocation
    # ----------------------
    def get_allocation_proportions(self, G, history, day: int, sim_days: int):
        self.global_gnn.eval()
        self.prior_booster.eval()
        with torch.no_grad():
            context = self._get_context(G, day, sim_days)

        alphas, betas = _accumulate_priors(history, self.num_blocks)

        # Boosts on device, pulled to CPU for numpy Beta sampling
        boosts = self.prior_booster(context)
        alphas = np.maximum(alphas + boosts[:, 0].detach().cpu().numpy(), 1e-6)
        betas  = np.maximum(betas  + boosts[:, 1].detach().cpu().numpy(), 1e-6)

        rates = np.random.beta(alphas, betas)
        return (
            rates / rates.sum()
            if rates.sum() > 0
            else np.ones(self.num_blocks) / self.num_blocks
        )

    # ----------------------
    # Learning
    # ----------------------
    def update(self, G, day: int, sim_days: int, daily_results, history=None):
        """Update global GNN and PriorBooster via Beta-Binomial NLL.

        Mirrors the LocalGNTS fix: base_alphas/base_betas are accumulated
        from the full test buffer so the gradient is computed against the
        empirical posterior rather than a flat prior.
        """
        self.global_gnn.train()
        self.prior_booster.train()
        self.optimizer.zero_grad()

        context = self._get_context(G, day, sim_days)

        base_alphas, base_betas = _accumulate_priors(history, self.num_blocks)

        # Loss scalar lives on device so .backward() propagates correctly
        loss  = torch.tensor(0.0, device=self.device)
        valid = 0

        for i in range(self.num_blocks):
            if i >= len(daily_results) or torch.all(context[i] == 0):
                continue

            boosts = self.prior_booster(context[i])

            # Wrap CPU numpy scalars as device tensors before arithmetic
            alpha = (
                torch.tensor(base_alphas[i], dtype=torch.float, device=self.device)
                + boosts[0]
            )
            beta = (
                torch.tensor(base_betas[i], dtype=torch.float, device=self.device)
                + boosts[1]
            )

            s = float(daily_results[i]['positive'])
            f = float(daily_results[i]['negative'])

            nll = -(
                torch.lgamma(alpha + s) + torch.lgamma(beta + f)
                - torch.lgamma(alpha + beta + s + f)
                - (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta))
            )

            if not torch.isnan(nll):
                loss  += nll
                valid += 1

        if valid > 0:
            loss = loss / valid
            loss.backward()
            self.optimizer.step()
            return loss.item()
        return 0.0

    # ----------------------
    # Persistence
    # ----------------------
    def save_model(self, path: str):
        # Move to CPU before serialising — checkpoints must be portable
        torch.save(
            {
                'global_gnn':    self.global_gnn.cpu().state_dict(),
                'prior_booster': self.prior_booster.cpu().state_dict(),
            },
            path,
        )
        # Restore to original device after saving
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