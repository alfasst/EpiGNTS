# strategies.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import random
from collections import OrderedDict, deque
from itertools import chain

# --- Heuristic Strategies (Unchanged) ---
def uniform_allocation(num_blocks, kits_per_day, **kwargs):
    allocations = np.full(num_blocks, kits_per_day // num_blocks); rem_kits = kits_per_day % num_blocks
    allocations[:rem_kits] += 1; return allocations

def random_allocation(num_blocks, kits_per_day, **kwargs):
    allocations = np.zeros(num_blocks)
    for _ in range(kits_per_day): allocations[random.randint(0, num_blocks - 1)] += 1
    return allocations.astype(int)

def proportional_allocation(num_blocks, kits_per_day, current_counts, **kwargs):
    i_plus_q = np.array([counts.get('I', 0) + counts.get('Q', 0) for counts in current_counts])
    total_i_plus_q = np.sum(i_plus_q)
    if total_i_plus_q == 0: return uniform_allocation(num_blocks, kits_per_day)
    proportions = i_plus_q / total_i_plus_q
    allocations = np.floor(proportions * kits_per_day).astype(int)
    rem_kits = kits_per_day - np.sum(allocations)
    if rem_kits > 0:
        residuals = (proportions * kits_per_day) - allocations
        for i in np.argsort(residuals)[-rem_kits:]: allocations[i] += 1
    return allocations

# --- Classic MAB Algorithms (Unchanged) ---
class BetaBinomialMAB:
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks; self.alphas = np.ones(num_blocks); self.betas = np.ones(num_blocks)
    def update_priors(self, historical_test_results):
        self.alphas = np.ones(self.num_blocks); self.betas = np.ones(self.num_blocks)
        for daily_result in historical_test_results:
            for i in range(self.num_blocks):
                self.alphas[i] += daily_result[i]['positive']; self.betas[i] += daily_result[i]['negative']
    def select_arm(self):
        samples = np.random.beta(self.alphas, self.betas); return np.argmax(samples)

class GammaPoissonMAB:
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks; self.shapes = np.ones(num_blocks); self.rates = np.ones(num_blocks)
    def update_priors(self, historical_test_results, lookback_days):
        self.shapes = np.ones(self.num_blocks); self.rates = np.ones(self.num_blocks)
        for daily_result in historical_test_results:
            for i in range(self.num_blocks): self.shapes[i] += daily_result[i]['positive']
        self.rates += lookback_days
    def select_arm(self):
        samples = np.random.gamma(self.shapes, 1.0 / self.rates); return np.argmax(samples)

# --- NEW: LocalGNTS Agent ---
class LocalGraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LocalGraphSAGE, self).__init__(); self.conv1=SAGEConv(input_dim,hidden_dim); self.conv2=SAGEConv(hidden_dim,output_dim)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index)); x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index); return x.mean(dim=0)

class PriorBooster(nn.Module):
    """A small MLP to predict boosts for alpha and beta priors."""
    def __init__(self, input_dim):
        super(PriorBooster, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2) # Outputs two values: delta_alpha, delta_beta
        )
    def forward(self, x):
        boosts = self.net(x)
        # Use softplus to ensure boosts are non-negative
        return F.softplus(boosts)

class LocalGNTS:
    def __init__(self, G, num_blocks, gnn_out_dim, context_dim, weight_decay):
        self.G = G; self.num_blocks = num_blocks
        
        self.local_gnns = nn.ModuleList([LocalGraphSAGE(4, 16, gnn_out_dim) for _ in range(num_blocks)])
        self.prior_boosters = nn.ModuleList([PriorBooster(context_dim) for _ in range(num_blocks)])
        
        all_params = chain(self.local_gnns.parameters(), self.prior_boosters.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=0.005, weight_decay=weight_decay)
        print(f"🚀 LocalGNTS Agent initialized. Context dim={context_dim}.")

    def _get_live_context(self, G, day, simulation_days):
        """Computes the context for each block with gradients enabled."""
        contexts = []
        for i in range(self.num_blocks):
            subgraph_nodes = [n for n, data in G.nodes(data=True) if data['block_id'] == i]
            subgraph = G.subgraph(subgraph_nodes)
            node_map = {node: j for j, node in enumerate(subgraph.nodes())}
            edges = list(subgraph.edges())
            local_edge_index = torch.tensor([[node_map.get(u, -1), node_map.get(v, -1)] for u, v in edges], dtype=torch.long).t().contiguous() if edges else torch.empty(2, 0, dtype=torch.long)
            local_node_features = torch.zeros(len(subgraph_nodes), 4)
            for j, node in enumerate(subgraph.nodes()):
                state = G.nodes[node]['state']
                if state == 'I': local_node_features[j, 0] = 1
                elif state == 'Q': local_node_features[j, 1] = 1
                elif state == 'R': local_node_features[j, 2] = 1
                else: local_node_features[j, 3] = 1
            local_data = Data(x=local_node_features, edge_index=local_edge_index)
            local_embedding = self.local_gnns[i](local_data)
            
            time_feature = torch.tensor([day / simulation_days], dtype=torch.float)
            contexts.append(torch.cat([local_embedding, time_feature]))
        return torch.stack(contexts)

    def get_allocation_proportions(self, G, historical_test_results, day, simulation_days):
        """Get allocation weights by sampling from the GNN-informed posterior."""
        self.local_gnns.eval(); self.prior_boosters.eval()
        with torch.no_grad():
            context = self._get_live_context(G, day, simulation_days)
        
        # 1. Calculate base priors from history
        base_alphas = np.ones(self.num_blocks); base_betas = np.ones(self.num_blocks)
        for daily_result in historical_test_results:
            for i in range(self.num_blocks):
                base_alphas[i] += daily_result[i]['positive']
                base_betas[i] += daily_result[i]['negative']
        
        # 2. Get boosts from the neural network
        boosts = torch.stack([self.prior_boosters[i](context[i]) for i in range(self.num_blocks)])
        delta_alphas = boosts[:, 0].detach().numpy()
        delta_betas = boosts[:, 1].detach().numpy()
        
        # 3. Combine and sample
        final_alphas = base_alphas + delta_alphas
        final_betas = base_betas + delta_betas
        
        sampled_hit_rates = np.random.beta(final_alphas, final_betas)
        return sampled_hit_rates

    def update(self, G, day, simulation_days, daily_test_results):
        """Update the GNNs and Prior Boosters using negative log-likelihood."""
        self.local_gnns.train(); self.prior_boosters.train()
        self.optimizer.zero_grad()
        
        context = self._get_live_context(G, day, simulation_days)
        
        base_alphas = np.ones(self.num_blocks); base_betas = np.ones(self.num_blocks)
        # Note: The history used for update should not include the current day's results
        # This is handled by the buffer management in simulation.py
        
        total_loss = 0
        for i in range(self.num_blocks):
            boosts = self.prior_boosters[i](context[i])
            alpha = torch.tensor(base_alphas[i]) + boosts[0]
            beta = torch.tensor(base_betas[i]) + boosts[1]
            
            successes = float(daily_test_results[i]['positive'])
            failures = float(daily_test_results[i]['negative'])
            
            # Negative Log-Likelihood of Beta-Binomial
            # We use torch.lgamma for the log of the Beta function: log(B(a,b)) = lgamma(a) + lgamma(b) - lgamma(a+b)
            log_beta_posterior = torch.lgamma(alpha + successes) + torch.lgamma(beta + failures) - torch.lgamma(alpha + beta + successes + failures)
            log_beta_prior = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
            
            # We want to maximize the likelihood, so we minimize the negative log-likelihood
            nll = -(log_beta_posterior - log_beta_prior)
            total_loss += nll
        
        if not torch.isnan(total_loss):
            total_loss.backward()
            self.optimizer.step()

def average_gnts_bandits(bandits, master_bandit_template):
    print(" consolidating trained bandits into a master model...")
    for module_name in ['local_gnns', 'prior_boosters']:
        module_list = getattr(bandits[0], module_name)
        for i in range(len(module_list)):
            avg_dict = OrderedDict()
            for k in module_list[i].state_dict().keys():
                avg_dict[k] = torch.stack([getattr(b, module_name)[i].state_dict()[k] for b in bandits]).mean(dim=0)
            getattr(master_bandit_template, module_name)[i].load_state_dict(avg_dict)
    print("✅ Master bandit created.")
    return master_bandit_template

