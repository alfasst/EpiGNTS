# gnts_vax.py
# ---------------------------------------------------------------------------
# Vaccine-adapted GNTS agent variants.
# Mirrors the structure of gnts.py with the following changes:
#
#   1. Node feature encoding uses build_node_features_vax() from netepi_vax
#      (4 buckets: S+E, V, I+A, R+Q) instead of the test-kit encoding.
#
#   2. Reward fields renamed:
#        'positive' -> 'success'  (infections averted / I-count reduction)
#        'negative' -> 'failure'  (infections that still occurred)
#      The Beta-Binomial NLL objective and Thompson sampling are mathematically
#      identical — only the semantic interpretation of the reward changes.
#
#   3. LocalGNTSVax and GlobalGNTSVax are standalone classes — they do NOT
#      inherit from or reference LocalGNTS / GlobalGNTS in gnts.py.
#      This keeps the two workflows fully independent.
#
#   4. average_gnts_vax_bandits() is a renamed copy of average_gnts_bandits()
#      that works with LocalGNTSVax / GlobalGNTSVax types.
#
# Public API (for simulation_vax.py):
#   LocalGNTSVax
#   GlobalGNTSVax
#   average_gnts_vax_bandits
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from collections import OrderedDict
from itertools import chain
import numpy as np

from netepi_vax import build_node_features_vax


# ---------------------------------------------------------------------------
# Shared neural network modules
# (GraphSAGEEncoder and PriorBooster are structurally identical to gnts.py;
#  duplicated here so gnts_vax.py has zero dependency on gnts.py)
# ---------------------------------------------------------------------------

class GraphSAGEEncoder(nn.Module):
    """
    Two-layer GraphSAGE returning per-node embeddings (N, out_dim).
    Pooling is done outside this module.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x                        # (N, output_dim)


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
# Shared training objective
# Identical math to gnts.py — Beta-Binomial NLL.
# ---------------------------------------------------------------------------

def _beta_binomial_nll(alpha, beta_param, successes, failures):
    """
    Negative log-likelihood of the Beta-Binomial update.
    'successes' = infection-reduction signal (was 'positive' in gnts.py)
    'failures'  = infections that still occurred (was 'negative' in gnts.py)
    """
    log_post = (torch.lgamma(alpha + successes)
                + torch.lgamma(beta_param + failures)
                - torch.lgamma(alpha + beta_param + successes + failures))
    log_prior = (torch.lgamma(alpha)
                 + torch.lgamma(beta_param)
                 - torch.lgamma(alpha + beta_param))
    return -(log_post - log_prior)


# ---------------------------------------------------------------------------
# Shared allocation logic
# Reads 'success' / 'failure' instead of 'positive' / 'negative'.
# ---------------------------------------------------------------------------

def _proportions_from_context(context, prior_boosters, num_blocks,
                               historical_vax_results):
    """
    Given a (num_blocks, context_dim) tensor and prior_boosters ModuleList,
    return a (num_blocks,) numpy array of sampled allocation proportions
    via Thompson sampling over the Beta posterior.

    historical_vax_results : deque of daily result lists.
        Each entry is a list[dict] of length num_blocks.
        Each dict has keys 'success' and 'failure'.
    """
    base_alphas = np.ones(num_blocks)
    base_betas  = np.ones(num_blocks)

    for daily_result in historical_vax_results:
        for i in range(num_blocks):
            base_alphas[i] += daily_result[i]['success']
            base_betas[i]  += daily_result[i]['failure']

    boosts       = torch.stack([prior_boosters[i](context[i])
                                for i in range(num_blocks)])
    final_alphas = base_alphas + boosts[:, 0].detach().numpy()
    final_betas  = base_betas  + boosts[:, 1].detach().numpy()

    samples = np.random.beta(final_alphas, final_betas)

    # Normalise to proportions summing to 1
    total = samples.sum()
    if total > 0:
        return samples / total
    return np.ones(num_blocks) / num_blocks       # fallback: uniform


def _compute_loss(context, prior_boosters, num_blocks, daily_vax_results):
    """
    Compute summed Beta-Binomial NLL loss across all blocks for one day.

    daily_vax_results : list[dict] of length num_blocks,
        each dict with keys 'success' and 'failure'.
    """
    total_loss = torch.tensor(0.0)
    for i in range(num_blocks):
        boosts    = prior_boosters[i](context[i])
        alpha     = torch.tensor(1.0) + boosts[0]
        beta_p    = torch.tensor(1.0) + boosts[1]
        successes = float(daily_vax_results[i]['success'])
        failures  = float(daily_vax_results[i]['failure'])
        total_loss = total_loss + _beta_binomial_nll(
            alpha, beta_p, successes, failures
        )
    return total_loss


# ---------------------------------------------------------------------------
# LocalGNTSVax
# ---------------------------------------------------------------------------

class LocalGNTSVax:
    """
    Vaccine-workflow variant of LocalGNTS.

    One independent GraphSAGE per block.
    Block embedding = mean pool of that block's node embeddings.
    Context = [block_embedding | time_fraction]  (dim = gnn_out_dim + 1)

    Node features use the vaccine 4-bucket encoding from netepi_vax:
        col 0 — S+E   (susceptible / exposed)
        col 1 — V     (vaccinated)
        col 2 — I+A   (infectious)
        col 3 — R+Q   (recovered / removed)

    Reward signal uses 'success' / 'failure' keys (lagged infection-reduction
    proxy computed in vaccination.py).
    """

    def __init__(self, sim_graph, num_blocks, gnn_out_dim, context_dim,
                 weight_decay):
        self.sim_graph  = sim_graph
        self.num_blocks = num_blocks

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
        print(f"💉 LocalGNTSVax initialised. "
              f"blocks={num_blocks}  context_dim={context_dim}")

    # ------------------------------------------------------------------
    # Graph pre-computation (identical logic to LocalGNTS)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Context building — uses vaccine node feature encoding
    # ------------------------------------------------------------------

    def _get_context(self, states, day, simulation_days):
        """
        Build (num_blocks, context_dim) context tensor.
        Uses build_node_features_vax() — vaccine 4-bucket encoding.
        """
        block_nodes  = self.sim_graph['block_nodes']
        time_feature = torch.tensor([day / simulation_days], dtype=torch.float)
        contexts     = []
        for i in range(self.num_blocks):
            nodes     = block_nodes[i]
            feats     = build_node_features_vax(states[nodes])
            node_embs = self.encoders[i](feats, self._block_edge_index[i])
            block_emb = node_embs.mean(dim=0)           # (gnn_out_dim,)
            contexts.append(torch.cat([block_emb, time_feature]))
        return torch.stack(contexts)                    # (num_blocks, context_dim)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_allocation_proportions(self, states, sim_graph,
                                   historical_vax_results, day, simulation_days):
        """
        Return (num_blocks,) numpy array of dose allocation proportions.
        Called each day before vaccination.
        """
        for m in (self.encoders, self.prior_boosters):
            for mod in m:
                mod.eval()
        with torch.no_grad():
            context = self._get_context(states, day, simulation_days)
        return _proportions_from_context(
            context, self.prior_boosters, self.num_blocks, historical_vax_results
        )

    def update(self, states, sim_graph, day, simulation_days, daily_vax_results):
        """
        One gradient step on the Beta-Binomial NLL using today's reward.
        Called after the lagged reward for this day becomes available.

        daily_vax_results : list[dict] with 'success'/'failure' per block.
        """
        for m in (self.encoders, self.prior_boosters):
            for mod in m:
                mod.train()
        self.optimizer.zero_grad()
        context    = self._get_context(states, day, simulation_days)
        total_loss = _compute_loss(
            context, self.prior_boosters, self.num_blocks, daily_vax_results
        )
        if not torch.isnan(total_loss):
            total_loss.backward()
            self.optimizer.step()


# ---------------------------------------------------------------------------
# GlobalGNTSVax
# ---------------------------------------------------------------------------

class GlobalGNTSVax:
    """
    Vaccine-workflow variant of GlobalGNTS.

    One shared GraphSAGE sees the entire network.
    Block embedding = mean pool of global embedding rows belonging to
                      nodes of that block.
    Context = [block_embedding | time_fraction]  (dim = gnn_out_dim + 1)

    Same vaccine encoding and 'success'/'failure' reward as LocalGNTSVax.
    """

    def __init__(self, sim_graph, num_blocks, gnn_out_dim, context_dim,
                 weight_decay):
        self.sim_graph  = sim_graph
        self.num_blocks = num_blocks

        self._global_edge_index = self._precompute_global_edge_index(sim_graph)

        self.encoder        = GraphSAGEEncoder(4, 16, gnn_out_dim)
        self.prior_boosters = nn.ModuleList(
            [PriorBooster(context_dim) for _ in range(num_blocks)]
        )

        all_params     = chain(self.encoder.parameters(),
                               self.prior_boosters.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=0.005,
                                          weight_decay=weight_decay)
        print(f"🌐 GlobalGNTSVax initialised. "
              f"blocks={num_blocks}  context_dim={context_dim}")

    # ------------------------------------------------------------------
    # Graph pre-computation (identical logic to GlobalGNTS)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Context building — uses vaccine node feature encoding
    # ------------------------------------------------------------------

    def _get_context(self, states, day, simulation_days):
        """
        1. Build global node features using vaccine 4-bucket encoding.
        2. Single forward pass through the shared encoder.
        3. Mean-pool per block.
        Returns (num_blocks, context_dim) tensor.
        """
        block_nodes  = self.sim_graph['block_nodes']
        time_feature = torch.tensor([day / simulation_days], dtype=torch.float)

        global_feats = build_node_features_vax(states)     # (n_nodes, 4)
        global_embs  = self.encoder(
            global_feats, self._global_edge_index
        )                                                   # (n_nodes, gnn_out_dim)

        contexts = []
        for i in range(self.num_blocks):
            nodes     = block_nodes[i]
            block_emb = global_embs[nodes].mean(dim=0)     # (gnn_out_dim,)
            contexts.append(torch.cat([block_emb, time_feature]))
        return torch.stack(contexts)                        # (num_blocks, context_dim)

    # ------------------------------------------------------------------
    # Public interface (identical signature to LocalGNTSVax)
    # ------------------------------------------------------------------

    def get_allocation_proportions(self, states, sim_graph,
                                   historical_vax_results, day, simulation_days):
        """
        Return (num_blocks,) numpy array of dose allocation proportions.
        """
        self.encoder.eval()
        for mod in self.prior_boosters:
            mod.eval()
        with torch.no_grad():
            context = self._get_context(states, day, simulation_days)
        return _proportions_from_context(
            context, self.prior_boosters, self.num_blocks, historical_vax_results
        )

    def update(self, states, sim_graph, day, simulation_days, daily_vax_results):
        """
        One gradient step using today's lagged reward.
        """
        self.encoder.train()
        for mod in self.prior_boosters:
            mod.train()
        self.optimizer.zero_grad()
        context    = self._get_context(states, day, simulation_days)
        total_loss = _compute_loss(
            context, self.prior_boosters, self.num_blocks, daily_vax_results
        )
        if not torch.isnan(total_loss):
            total_loss.backward()
            self.optimizer.step()


# ---------------------------------------------------------------------------
# Model averaging — vaccine agent variants
# ---------------------------------------------------------------------------

def average_gnts_vax_bandits(bandits, master_template):
    """
    Federated-average neural network weights across a list of trained
    vaccine agents into master_template.

    Handles both LocalGNTSVax (has 'encoders' ModuleList) and
    GlobalGNTSVax (has single 'encoder' + 'prior_boosters').

    Parameters
    ----------
    bandits         : list[LocalGNTSVax | GlobalGNTSVax]
    master_template : freshly constructed agent of the same class

    Returns
    -------
    master_template with averaged weights loaded in-place.
    """
    print("  Consolidating trained vaccine agents into master model...")
    agent_type = type(bandits[0])

    if agent_type is LocalGNTSVax:
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

    elif agent_type is GlobalGNTSVax:
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
        raise TypeError(
            f"average_gnts_vax_bandits: unknown agent type {agent_type}. "
            f"Expected LocalGNTSVax or GlobalGNTSVax."
        )

    print("✅ Master vaccine agent created.")
    return master_template
