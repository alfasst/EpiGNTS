# gnts.py
# ---------------------------------------------------------------------------
# Contains both GNTS agent variants:
#
#   LocalGNTS  — one GNN per block, each sees only its block's subgraph.
#                Block embedding = mean-pooled node embeddings of that subgraph.
#
#   GlobalGNTS — one shared GNN sees the entire network.
#                Block embedding = mean-pooled node embeddings of nodes that
#                belong to that block, extracted from the global embedding matrix.
#
# Both agents share:
#   - The same 4-dim node feature encoding  (I->0, Q->1, R->2, else->3)
#   - The same PriorBooster MLP
#   - The same Beta-Binomial NLL training objective
#   - The same get_allocation_proportions / update interface
#   - average_gnts_bandits() for model averaging after parallel training runs
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from collections import OrderedDict
from itertools import chain
import numpy as np

from network_epidemic import I, Q, R, S, E, A

# ---------------------------------------------------------------------------
# Node feature encoding — shared by both agents
# I->0, Q->1, R->2, everything else (S, E, A) -> 3
# ---------------------------------------------------------------------------
_FEAT_IDX  = {I: 0, Q: 1, R: 2}
_OTHER_STATES = np.array([S, E, A], dtype=np.int8)


def _build_node_features(states_subset):
    """
    Given a 1-D int8 array of states for a set of nodes,
    return a (N, 4) float32 torch tensor with one-hot encoding.
    """
    n = len(states_subset)
    feats = torch.zeros(n, 4)
    for state_int, col in _FEAT_IDX.items():
        feats[states_subset == state_int, col] = 1.0
    other_mask = np.isin(states_subset, _OTHER_STATES)
    feats[other_mask, 3] = 1.0
    return feats


# ---------------------------------------------------------------------------
# Shared neural network modules
# ---------------------------------------------------------------------------

class GraphSAGEEncoder(nn.Module):
    """
    Two-layer GraphSAGE that returns a per-node embedding matrix (N, out_dim).
    Used by both Local and Global variants; pooling is done outside.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x                    # (N, output_dim) — no pooling here


class PriorBooster(nn.Module):
    """Small MLP: context -> (delta_alpha, delta_beta), both non-negative."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return F.softplus(self.net(x))


# ---------------------------------------------------------------------------
# Shared training objective & allocation logic
# ---------------------------------------------------------------------------

def _beta_binomial_nll(alpha, beta, successes, failures):
    """Negative log-likelihood of the Beta-Binomial update."""
    log_post = (torch.lgamma(alpha + successes)
                + torch.lgamma(beta + failures)
                - torch.lgamma(alpha + beta + successes + failures))
    log_prior = (torch.lgamma(alpha)
                 + torch.lgamma(beta)
                 - torch.lgamma(alpha + beta))
    return -(log_post - log_prior)


def _proportions_from_context(context, prior_boosters, num_blocks,
                               historical_test_results):
    """
    Given a (num_blocks, context_dim) tensor and prior_boosters ModuleList,
    return a (num_blocks,) numpy array of sampled hit-rate proportions.
    """
    base_alphas = np.ones(num_blocks)
    base_betas  = np.ones(num_blocks)
    for daily_result in historical_test_results:
        for i in range(num_blocks):
            base_alphas[i] += daily_result[i]['positive']
            base_betas[i]  += daily_result[i]['negative']

    boosts       = torch.stack([prior_boosters[i](context[i])
                                for i in range(num_blocks)])
    final_alphas = base_alphas + boosts[:, 0].detach().numpy()
    final_betas  = base_betas  + boosts[:, 1].detach().numpy()
    return np.random.beta(final_alphas, final_betas)


def _compute_loss(context, prior_boosters, num_blocks, daily_test_results):
    """Compute summed NLL loss across all blocks."""
    total_loss = torch.tensor(0.0)
    for i in range(num_blocks):
        boosts    = prior_boosters[i](context[i])
        alpha     = torch.tensor(1.0) + boosts[0]
        beta      = torch.tensor(1.0) + boosts[1]
        successes = float(daily_test_results[i]['positive'])
        failures  = float(daily_test_results[i]['negative'])
        total_loss = total_loss + _beta_binomial_nll(alpha, beta,
                                                     successes, failures)
    return total_loss


# ---------------------------------------------------------------------------
# LocalGNTS
# ---------------------------------------------------------------------------

class LocalGNTS:
    """
    One independent GraphSAGE per block.
    Block embedding = mean pool of that block's node embeddings.
    Context = [block_embedding | time_fraction]  (dim = gnn_out_dim + 1)
    """

    def __init__(self, sim_graph, num_blocks, gnn_out_dim, context_dim, weight_decay):
        self.sim_graph  = sim_graph
        self.num_blocks = num_blocks

        # Precompute per-block edge indices (topology fixed for entire run)
        self._block_edge_index = self._precompute_block_edge_indices(sim_graph)

        self.encoders       = nn.ModuleList(
            [GraphSAGEEncoder(4, 16, gnn_out_dim) for _ in range(num_blocks)]
        )
        self.prior_boosters = nn.ModuleList(
            [PriorBooster(context_dim) for _ in range(num_blocks)]
        )

        all_params     = chain(self.encoders.parameters(),
                               self.prior_boosters.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=0.005,
                                          weight_decay=weight_decay)
        print(f"🚀 LocalGNTS initialised. "
              f"blocks={num_blocks}  context_dim={context_dim}")

    @staticmethod
    def _precompute_block_edge_indices(sim_graph):
        """Build locally re-indexed edge_index tensors per block."""
        adj_lists   = sim_graph['adj_lists']
        block_nodes = sim_graph['block_nodes']
        num_blocks  = sim_graph['num_blocks']
        edge_indices = []
        for b in range(num_blocks):
            nodes    = block_nodes[b]
            node_set = set(nodes.tolist())
            local_id = {int(gid): lid for lid, gid in enumerate(nodes.tolist())}
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

    def _get_context(self, states, day, simulation_days):
        """(num_blocks, context_dim) tensor — one embedding per block."""
        block_nodes  = self.sim_graph['block_nodes']
        time_feature = torch.tensor([day / simulation_days], dtype=torch.float)
        contexts     = []
        for i in range(self.num_blocks):
            nodes        = block_nodes[i]
            feats        = _build_node_features(states[nodes])
            node_embs    = self.encoders[i](feats, self._block_edge_index[i])
            block_emb    = node_embs.mean(dim=0)          # (gnn_out_dim,)
            contexts.append(torch.cat([block_emb, time_feature]))
        return torch.stack(contexts)                      # (num_blocks, context_dim)

    # Public interface (identical signature for both agents)

    def get_allocation_proportions(self, states, sim_graph,
                                   historical_test_results, day, simulation_days):
        for m in (self.encoders, self.prior_boosters):
            for mod in m: mod.eval()
        with torch.no_grad():
            context = self._get_context(states, day, simulation_days)
        return _proportions_from_context(context, self.prior_boosters,
                                         self.num_blocks, historical_test_results)

    def update(self, states, sim_graph, day, simulation_days, daily_test_results):
        for m in (self.encoders, self.prior_boosters):
            for mod in m: mod.train()
        self.optimizer.zero_grad()
        context    = self._get_context(states, day, simulation_days)
        total_loss = _compute_loss(context, self.prior_boosters,
                                   self.num_blocks, daily_test_results)
        if not torch.isnan(total_loss):
            total_loss.backward()
            self.optimizer.step()


# ---------------------------------------------------------------------------
# GlobalGNTS
# ---------------------------------------------------------------------------

class GlobalGNTS:
    """
    One shared GraphSAGE sees the entire network.
    Block embedding = mean pool of the global embedding rows that belong
                      to nodes of that block.
    Context = [block_embedding | time_fraction]  (dim = gnn_out_dim + 1)

    The global edge_index is precomputed once at construction.
    """

    def __init__(self, sim_graph, num_blocks, gnn_out_dim, context_dim, weight_decay):
        self.sim_graph  = sim_graph
        self.num_blocks = num_blocks

        # Precompute global edge_index (COO format, shape (2, E))
        self._global_edge_index = self._precompute_global_edge_index(sim_graph)

        self.encoder        = GraphSAGEEncoder(4, 16, gnn_out_dim)
        self.prior_boosters = nn.ModuleList(
            [PriorBooster(context_dim) for _ in range(num_blocks)]
        )

        all_params     = chain(self.encoder.parameters(),
                               self.prior_boosters.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=0.005,
                                          weight_decay=weight_decay)
        print(f"🌐 GlobalGNTS initialised. "
              f"blocks={num_blocks}  context_dim={context_dim}")

    @staticmethod
    def _precompute_global_edge_index(sim_graph):
        """Build a single COO edge_index for the whole graph."""
        adj_lists = sim_graph['adj_lists']
        n_nodes   = sim_graph['n_nodes']
        src, dst  = [], []
        for node in range(n_nodes):
            for nb in adj_lists[node]:
                src.append(node)
                dst.append(int(nb))
        if src:
            return torch.tensor([src, dst], dtype=torch.long)
        return torch.empty((2, 0), dtype=torch.long)

    def _get_context(self, states, day, simulation_days):
        """
        1. Build global node features from full state array.
        2. Run one forward pass through the shared encoder.
        3. For each block, mean-pool the rows belonging to that block.
        Returns (num_blocks, context_dim) tensor.
        """
        block_nodes  = self.sim_graph['block_nodes']
        time_feature = torch.tensor([day / simulation_days], dtype=torch.float)

        # Global feature matrix (n_nodes, 4)
        global_feats = _build_node_features(states)

        # Single GNN forward pass over the whole graph
        global_embs  = self.encoder(global_feats, self._global_edge_index)
        # global_embs: (n_nodes, gnn_out_dim)

        contexts = []
        for i in range(self.num_blocks):
            nodes     = block_nodes[i]                    # np.ndarray of global ids
            block_emb = global_embs[nodes].mean(dim=0)   # (gnn_out_dim,)
            contexts.append(torch.cat([block_emb, time_feature]))
        return torch.stack(contexts)                      # (num_blocks, context_dim)

    # Public interface (identical signature to LocalGNTS)

    def get_allocation_proportions(self, states, sim_graph,
                                   historical_test_results, day, simulation_days):
        self.encoder.eval()
        for mod in self.prior_boosters: mod.eval()
        with torch.no_grad():
            context = self._get_context(states, day, simulation_days)
        return _proportions_from_context(context, self.prior_boosters,
                                         self.num_blocks, historical_test_results)

    def update(self, states, sim_graph, day, simulation_days, daily_test_results):
        self.encoder.train()
        for mod in self.prior_boosters: mod.train()
        self.optimizer.zero_grad()
        context    = self._get_context(states, day, simulation_days)
        total_loss = _compute_loss(context, self.prior_boosters,
                                   self.num_blocks, daily_test_results)
        if not torch.isnan(total_loss):
            total_loss.backward()
            self.optimizer.step()


# ---------------------------------------------------------------------------
# Model averaging — works for both agent types
# ---------------------------------------------------------------------------

def average_gnts_bandits(bandits, master_template):
    """
    Federated-average the neural network weights across a list of trained
    agents into master_template (must be the same class as bandits[0]).

    Handles both LocalGNTS (has 'encoders' ModuleList) and
    GlobalGNTS (has 'encoder' single module + 'prior_boosters').
    """
    print("  Consolidating trained agents into master model...")
    agent_type = type(bandits[0])

    if agent_type is LocalGNTS:
        for module_name in ['encoders', 'prior_boosters']:
            module_list = getattr(bandits[0], module_name)
            for i in range(len(module_list)):
                avg_dict = OrderedDict()
                for k in module_list[i].state_dict().keys():
                    avg_dict[k] = torch.stack(
                        [getattr(b, module_name)[i].state_dict()[k]
                         for b in bandits]
                    ).float().mean(dim=0)
                getattr(master_template, module_name)[i].load_state_dict(avg_dict)

    elif agent_type is GlobalGNTS:
        # Average the single shared encoder
        avg_enc = OrderedDict()
        for k in bandits[0].encoder.state_dict().keys():
            avg_enc[k] = torch.stack(
                [b.encoder.state_dict()[k] for b in bandits]
            ).float().mean(dim=0)
        master_template.encoder.load_state_dict(avg_enc)

        # Average per-block prior boosters
        num_blocks = len(bandits[0].prior_boosters)
        for i in range(num_blocks):
            avg_dict = OrderedDict()
            for k in bandits[0].prior_boosters[i].state_dict().keys():
                avg_dict[k] = torch.stack(
                    [b.prior_boosters[i].state_dict()[k] for b in bandits]
                ).float().mean(dim=0)
            master_template.prior_boosters[i].load_state_dict(avg_dict)

    else:
        raise TypeError(f"average_gnts_bandits: unknown agent type {agent_type}")

    print("✅ Master agent created.")
    return master_template