# gnts.py (UPDATED)
# Restores immediate-feedback GNTS behavior with explicit exploration
# while keeping vectorization and pooling intact.

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
# Device
# --------------------------------------------------

def _resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device('cpu')
    major, minor = torch.cuda.get_device_capability(0)
    if major < 7:
        return torch.device('cpu')
    return torch.device('cuda')

DEVICE = _resolve_device()

# --------------------------------------------------
# GNN blocks
# --------------------------------------------------

class GraphSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        if edge_index.numel() == 0:
            return torch.zeros((x.size(0), config.GNN_OUTPUT_DIM), device=x.device)
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

def build_edge_index(G: nx.Graph, node_map: dict, device=DEVICE):
    edges = [(node_map[u], node_map[v]) for u, v in G.edges() if u in node_map and v in node_map]
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()


# --------------------------------------------------
# Local GNTS
# --------------------------------------------------

class LocalGNTS:
    def __init__(self, G, num_blocks, gnn_out_dim, context_dim, weight_decay, device=DEVICE):
        self.G = G
        self.num_blocks = num_blocks
        self.device = device

        self.local_gnns = nn.ModuleList([
            GraphSAGE(4, 16, gnn_out_dim) for _ in range(num_blocks)
        ]).to(device)

        self.prior_boosters = nn.ModuleList([
            PriorBooster(context_dim) for _ in range(num_blocks)
        ]).to(device)

        params = chain(self.local_gnns.parameters(), self.prior_boosters.parameters())
        self.optimizer = torch.optim.Adam(params, lr=0.005, weight_decay=weight_decay)

    # ----------------------
    # Context
    # ----------------------
    def _get_context(self, G, day, sim_days):
        contexts = []
        for bid in range(self.num_blocks):
            nodes = [n for n, d in G.nodes(data=True) if d.get('block_id') == bid]
            if not nodes:
                contexts.append(torch.zeros(config.LOCAL_AGENT_CONTEXT_DIM, device=self.device))
                continue

            sub = G.subgraph(nodes)
            node_map = {n: i for i, n in enumerate(sub.nodes())}

            x = torch.zeros((len(nodes), 4), device=self.device)
            for i, n in enumerate(sub.nodes()):
                s = G.nodes[n].get('state', 'S')
                idx = ['I', 'Q', 'R'].index(s) if s in ['I', 'Q', 'R'] else 3
                x[i, idx] = 1.0

            edge_index = build_edge_index(sub, node_map, self.device)
            emb = self.local_gnns[bid](Data(x=x, edge_index=edge_index))
            block_emb = emb.mean(dim=0) if emb.numel() > 0 else torch.zeros(config.GNN_OUTPUT_DIM, device=self.device)

            t = torch.tensor([day / sim_days], dtype=torch.float, device=self.device)
            contexts.append(torch.cat([block_emb, t]))

        return torch.stack(contexts)

    # ----------------------
    # Allocation (no history, with exploration)
    # ----------------------
    def get_allocation_proportions(self, G, day, sim_days):
        self.local_gnns.eval()
        self.prior_boosters.eval()
        with torch.no_grad():
            context = self._get_context(G, day, sim_days)

        base_alphas = np.ones(self.num_blocks, dtype=np.float32)
        base_betas = np.ones(self.num_blocks, dtype=np.float32)

        boosts = torch.stack([self.prior_boosters[i](context[i]) for i in range(self.num_blocks)])
        alphas = np.maximum(base_alphas + boosts[:, 0].detach().cpu().numpy(), 1e-6)
        betas  = np.maximum(base_betas  + boosts[:, 1].detach().cpu().numpy(), 1e-6)

        if np.random.rand() < config.EXPLORATION_BUDGET_FRACTION:
            rates = np.random.rand(self.num_blocks)
        else:
            rates = np.random.beta(alphas, betas)

        return rates / rates.sum() if rates.sum() > 0 else np.ones(self.num_blocks) / self.num_blocks

    def save_model(self, path: str):
        torch.save({
            'local_gnns': self.local_gnns.state_dict(),
            'prior_boosters': self.prior_boosters.state_dict(),
        }, path)


    def load_model(self, path: str):
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=self.device)
        self.local_gnns.load_state_dict(ckpt['local_gnns'])
        self.prior_boosters.load_state_dict(ckpt['prior_boosters'])
    
    # ----------------------
    # Update (no history)
    # ----------------------    
    def update(self, G, day, sim_days, daily_results):
        self.local_gnns.train()
        self.prior_boosters.train()
        self.optimizer.zero_grad()

        context = self._get_context(G, day, sim_days)

        base_alphas = np.ones(self.num_blocks, dtype=np.float32)
        base_betas = np.ones(self.num_blocks, dtype=np.float32)

        loss = torch.tensor(0.0, device=self.device)
        valid = 0

        for i in range(self.num_blocks):
            if i >= len(daily_results) or torch.all(context[i] == 0):
                continue

            boosts = self.prior_boosters[i](context[i])
            alpha = torch.tensor(base_alphas[i], device=self.device) + boosts[0]
            beta = torch.tensor(base_betas[i], device=self.device) + boosts[1]

            s = float(daily_results[i]['positive'])
            f = float(daily_results[i]['negative'])

            nll = -(torch.lgamma(alpha + s) + torch.lgamma(beta + f)
                    - torch.lgamma(alpha + beta + s + f)
                    - (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)))

            if not torch.isnan(nll):
                loss += nll
                valid += 1

        if valid > 0:
            (loss / valid).backward()
            self.optimizer.step()
            return (loss / valid).item()
        return 0.0


# --------------------------------------------------
# Global GNTS (same logic, no history)
# --------------------------------------------------

class GlobalGNTS:
    def __init__(self, G, num_blocks, gnn_out_dim, context_dim, weight_decay, device=DEVICE, precomputed=None):
        self.num_blocks = num_blocks
        self.device = device

        self.global_gnn = GraphSAGE(4, 16, gnn_out_dim).to(device)
        self.prior_booster = PriorBooster(context_dim).to(device)

        params = chain(self.global_gnn.parameters(), self.prior_booster.parameters())
        self.optimizer = torch.optim.Adam(params, lr=0.005, weight_decay=weight_decay)

        node_index, nodes, _, _, edge_index_t, block_id_t = precomputed

        self.node_map = node_index
        self.nodes = nodes
        self.N = len(nodes)
        self.edge_index_t = edge_index_t.to(device)

        raw_bid = block_id_t.to(device)
        self.valid_mask = raw_bid >= 0
        self.block_id_t = raw_bid.clamp(min=0)

    def _get_context(self, G, day, sim_days):
        x = torch.zeros((self.N, 4), device=self.device)
        for n, idx in self.node_map.items():
            s = G.nodes[n].get('state', 'S')
            feat = ['I', 'Q', 'R'].index(s) if s in ['I', 'Q', 'R'] else 3
            x[idx, feat] = 1.0

        node_emb = self.global_gnn(Data(x=x, edge_index=self.edge_index_t))

        if not self.valid_mask.all():
            node_emb = node_emb * self.valid_mask.unsqueeze(1).float()

        block_emb = scatter_mean(
            node_emb,
            self.block_id_t,
            dim=0,
            out=torch.zeros((self.num_blocks, config.GNN_OUTPUT_DIM), device=self.device)
        )

        t = torch.full((self.num_blocks, 1), day / sim_days, device=self.device)
        return torch.cat([block_emb, t], dim=1)

    def get_allocation_proportions(self, G, day, sim_days):
        self.global_gnn.eval()
        self.prior_booster.eval()
        with torch.no_grad():
            context = self._get_context(G, day, sim_days)

        base_alphas = np.ones(self.num_blocks, dtype=np.float32)
        base_betas = np.ones(self.num_blocks, dtype=np.float32)

        boosts = self.prior_booster(context)
        alphas = np.maximum(base_alphas + boosts[:, 0].detach().cpu().numpy(), 1e-6)
        betas  = np.maximum(base_betas  + boosts[:, 1].detach().cpu().numpy(), 1e-6)

        if np.random.rand() < config.EXPLORATION_BUDGET_FRACTION:
            rates = np.random.rand(self.num_blocks)
        else:
            rates = np.random.beta(alphas, betas)

        return rates / rates.sum() if rates.sum() > 0 else np.ones(self.num_blocks) / self.num_blocks

    def save_model(self, path: str):
        torch.save({
            'global_gnn': self.global_gnn.state_dict(),
            'prior_booster': self.prior_booster.state_dict(),
        }, path)


    def load_model(self, path: str):
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location=self.device)
        self.global_gnn.load_state_dict(ckpt['global_gnn'])
        self.prior_booster.load_state_dict(ckpt['prior_booster'])    
    
    
    def update(self, G, day, sim_days, daily_results):
        self.global_gnn.train()
        self.prior_booster.train()
        self.optimizer.zero_grad()

        context = self._get_context(G, day, sim_days)

        base_alphas = np.ones(self.num_blocks, dtype=np.float32)
        base_betas = np.ones(self.num_blocks, dtype=np.float32)

        loss = torch.tensor(0.0, device=self.device)
        valid = 0

        for i in range(self.num_blocks):
            if i >= len(daily_results) or torch.all(context[i] == 0):
                continue

            boosts = self.prior_booster(context[i])
            alpha = torch.tensor(base_alphas[i], device=self.device) + boosts[0]
            beta = torch.tensor(base_betas[i], device=self.device) + boosts[1]

            s = float(daily_results[i]['positive'])
            f = float(daily_results[i]['negative'])

            nll = -(torch.lgamma(alpha + s) + torch.lgamma(beta + f)
                    - torch.lgamma(alpha + beta + s + f)
                    - (torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)))

            if not torch.isnan(nll):
                loss += nll
                valid += 1

        if valid > 0:
            (loss / valid).backward()
            self.optimizer.step()
            return (loss / valid).item()
        return 0.0
