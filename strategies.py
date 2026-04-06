# strategies.py

import random
from itertools import chain
from collections import OrderedDict

import numpy as np

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