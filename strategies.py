# strategies.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
import random
from collections import OrderedDict
from itertools import chain

from network_epidemic import S, E, I, A, Q, R, STATE_NAMES

# ---------------------------------------------------------------------------
# Heuristic Strategies — unchanged logic, same interface
# ---------------------------------------------------------------------------

def uniform_allocation(num_blocks, kits_per_day, **kwargs):
    allocations = np.full(num_blocks, kits_per_day // num_blocks)
    rem_kits = kits_per_day % num_blocks
    allocations[:rem_kits] += 1
    return allocations

def random_allocation(num_blocks, kits_per_day, **kwargs):
    allocations = np.zeros(num_blocks, dtype=int)
    for _ in range(kits_per_day):
        allocations[random.randint(0, num_blocks - 1)] += 1
    return allocations

def proportional_allocation(num_blocks, kits_per_day, current_counts, **kwargs):
    i_plus_q = np.array([counts.get('I', 0) + counts.get('Q', 0) for counts in current_counts])
    total_i_plus_q = np.sum(i_plus_q)
    if total_i_plus_q == 0:
        return uniform_allocation(num_blocks, kits_per_day)
    proportions = i_plus_q / total_i_plus_q
    allocations = np.floor(proportions * kits_per_day).astype(int)
    rem_kits = kits_per_day - np.sum(allocations)
    if rem_kits > 0:
        residuals = (proportions * kits_per_day) - allocations
        for i in np.argsort(residuals)[-rem_kits:]:
            allocations[i] += 1
    return allocations


# ---------------------------------------------------------------------------
# Classic MAB Algorithms — unchanged
# ---------------------------------------------------------------------------

class BetaBinomialMAB:
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.alphas = np.ones(num_blocks)
        self.betas  = np.ones(num_blocks)

    def update_priors(self, historical_test_results):
        self.alphas = np.ones(self.num_blocks)
        self.betas  = np.ones(self.num_blocks)
        for daily_result in historical_test_results:
            for i in range(self.num_blocks):
                self.alphas[i] += daily_result[i]['positive']
                self.betas[i]  += daily_result[i]['negative']

    def select_arm(self):
        return int(np.argmax(np.random.beta(self.alphas, self.betas)))


class GammaPoissonMAB:
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.shapes = np.ones(num_blocks)
        self.rates  = np.ones(num_blocks)

    def update_priors(self, historical_test_results, lookback_days):
        self.shapes = np.ones(self.num_blocks)
        self.rates  = np.ones(self.num_blocks)
        for daily_result in historical_test_results:
            for i in range(self.num_blocks):
                self.shapes[i] += daily_result[i]['positive']
        self.rates += lookback_days

    def select_arm(self):
        return int(np.argmax(np.random.gamma(self.shapes, 1.0 / self.rates)))


# ---------------------------------------------------------------------------
# GNN modules — unchanged architectures
# ---------------------------------------------------------------------------

class LocalGraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x.mean(dim=0)


class PriorBooster(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),   # delta_alpha, delta_beta
        )

    def forward(self, x):
        return F.softplus(self.net(x))


# ---------------------------------------------------------------------------
# LocalGNTS — updated to use (states, sim_graph) instead of NetworkX G
# ---------------------------------------------------------------------------

# Feature index mapping for the 4-dim node feature vector.
# Old code: I->0, Q->1, R->2, else->3   (same mapping kept exactly)
_FEAT_IDX = {I: 0, Q: 1, R: 2}   # S/E/A all map to 3

class LocalGNTS:
    def __init__(self, sim_graph, num_blocks, gnn_out_dim, context_dim, weight_decay):
        self.sim_graph  = sim_graph
        self.num_blocks = num_blocks

        # Cache per-block edge_index tensors — topology never changes
        self._block_edge_index = self._precompute_edge_indices(sim_graph)

        self.local_gnns    = nn.ModuleList(
            [LocalGraphSAGE(4, 16, gnn_out_dim) for _ in range(num_blocks)]
        )
        self.prior_boosters = nn.ModuleList(
            [PriorBooster(context_dim) for _ in range(num_blocks)]
        )

        all_params = chain(self.local_gnns.parameters(),
                           self.prior_boosters.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=0.005,
                                          weight_decay=weight_decay)
        print(f"🚀 LocalGNTS Agent initialised. Context dim={context_dim}.")

    # ------------------------------------------------------------------
    # One-time topology cache
    # ------------------------------------------------------------------
    @staticmethod
    def _precompute_edge_indices(sim_graph):
        """
        Build a local (re-indexed 0..block_size-1) edge_index tensor
        for each block. Called once at construction; reused every day.
        """
        adj_lists   = sim_graph['adj_lists']
        block_nodes = sim_graph['block_nodes']
        num_blocks  = sim_graph['num_blocks']

        edge_indices = []
        for b in range(num_blocks):
            nodes    = block_nodes[b]           # np.ndarray of global node ids
            node_set = set(nodes.tolist())
            local_id = {gid: lid for lid, gid in enumerate(nodes.tolist())}

            src, dst = [], []
            for gid in nodes:
                for nb in adj_lists[gid]:
                    if nb in node_set:
                        src.append(local_id[int(gid)])
                        dst.append(local_id[int(nb)])

            if src:
                ei = torch.tensor([src, dst], dtype=torch.long)
            else:
                ei = torch.empty((2, 0), dtype=torch.long)
            edge_indices.append(ei)

        return edge_indices

    # ------------------------------------------------------------------
    # Context computation — only node features rebuild each day
    # ------------------------------------------------------------------
    def _get_live_context(self, states, day, simulation_days):
        """
        Build per-block context vectors.
        Uses precomputed edge indices; only recomputes node feature tensors.
        Feature encoding identical to old code: I->0, Q->1, R->2, else->3.
        """
        block_nodes = self.sim_graph['block_nodes']
        contexts    = []

        for i in range(self.num_blocks):
            nodes  = block_nodes[i]                         # np.ndarray, global ids
            n_nodes = len(nodes)
            feats  = torch.zeros(n_nodes, 4)

            block_states = states[nodes]                    # int8 array
            for feat_state, feat_col in _FEAT_IDX.items():
                mask = block_states == feat_state
                feats[mask, feat_col] = 1.0
            # everything else (S, E, A) -> column 3
            other_mask = ~np.isin(block_states, list(_FEAT_IDX.keys()))
            feats[other_mask, 3] = 1.0

            local_data      = Data(x=feats, edge_index=self._block_edge_index[i])
            local_embedding = self.local_gnns[i](local_data)

            time_feature = torch.tensor([day / simulation_days], dtype=torch.float)
            contexts.append(torch.cat([local_embedding, time_feature]))

        return torch.stack(contexts)

    # ------------------------------------------------------------------
    # Public interface — signatures updated to (states, sim_graph, ...)
    # ------------------------------------------------------------------

    def get_allocation_proportions(self, states, sim_graph,
                                   historical_test_results, day, simulation_days):
        self.local_gnns.eval()
        self.prior_boosters.eval()
        with torch.no_grad():
            context = self._get_live_context(states, day, simulation_days)

        base_alphas = np.ones(self.num_blocks)
        base_betas  = np.ones(self.num_blocks)
        for daily_result in historical_test_results:
            for i in range(self.num_blocks):
                base_alphas[i] += daily_result[i]['positive']
                base_betas[i]  += daily_result[i]['negative']

        boosts = torch.stack(
            [self.prior_boosters[i](context[i]) for i in range(self.num_blocks)]
        )
        delta_alphas = boosts[:, 0].detach().numpy()
        delta_betas  = boosts[:, 1].detach().numpy()

        final_alphas = base_alphas + delta_alphas
        final_betas  = base_betas  + delta_betas
        return np.random.beta(final_alphas, final_betas)

    def update(self, states, sim_graph, day, simulation_days, daily_test_results):
        self.local_gnns.train()
        self.prior_boosters.train()
        self.optimizer.zero_grad()

        context = self._get_live_context(states, day, simulation_days)

        base_alphas = np.ones(self.num_blocks)
        base_betas  = np.ones(self.num_blocks)

        total_loss = torch.tensor(0.0)
        for i in range(self.num_blocks):
            boosts = self.prior_boosters[i](context[i])
            alpha  = torch.tensor(base_alphas[i]) + boosts[0]
            beta   = torch.tensor(base_betas[i])  + boosts[1]

            successes = float(daily_test_results[i]['positive'])
            failures  = float(daily_test_results[i]['negative'])

            log_beta_posterior = (torch.lgamma(alpha + successes)
                                  + torch.lgamma(beta + failures)
                                  - torch.lgamma(alpha + beta + successes + failures))
            log_beta_prior = (torch.lgamma(alpha)
                              + torch.lgamma(beta)
                              - torch.lgamma(alpha + beta))
            nll = -(log_beta_posterior - log_beta_prior)
            total_loss = total_loss + nll

        if not torch.isnan(total_loss):
            total_loss.backward()
            self.optimizer.step()


# ---------------------------------------------------------------------------
# Model averaging — unchanged logic
# ---------------------------------------------------------------------------

def average_gnts_bandits(bandits, master_bandit_template):
    print(" consolidating trained bandits into a master model...")
    for module_name in ['local_gnns', 'prior_boosters']:
        module_list = getattr(bandits[0], module_name)
        for i in range(len(module_list)):
            avg_dict = OrderedDict()
            for k in module_list[i].state_dict().keys():
                avg_dict[k] = torch.stack(
                    [getattr(b, module_name)[i].state_dict()[k] for b in bandits]
                ).mean(dim=0)
            getattr(master_bandit_template, module_name)[i].load_state_dict(avg_dict)
    print("✅ Master bandit created.")
    return master_bandit_template