# strategies.py
# Lightweight strategy wrappers and non-GNTS baselines
# --------------------------------------------------
# This file intentionally contains ONLY:
# - Heuristic allocation strategies
# - Multi-Armed Bandit (MAB) baselines
#
# Core GNTS logic lives in gnts.py and MUST NOT be duplicated here.
#
# Changes vs previous version:
# - proportional_allocation weights by I + Q instead of I + A.
#   Q (confirmed quarantined) is a directly observable signal; A
#   (asymptomatic) is largely undetected, making it a noisier proxy.
# - GammaPoissonMAB removed.

import numpy as np


def uniform_allocation(num_blocks, kits, **kwargs):
    if num_blocks <= 0 or kits <= 0:
        return np.zeros(num_blocks, dtype=int)
    base = kits // num_blocks
    alloc = np.full(num_blocks, base, dtype=int)
    alloc[: kits - alloc.sum()] += 1
    return alloc


def random_allocation(num_blocks, kits, **kwargs):
    if num_blocks <= 0 or kits <= 0:
        return np.zeros(num_blocks, dtype=int)
    alloc = np.zeros(num_blocks, dtype=int)
    for _ in range(kits):
        alloc[np.random.randint(0, num_blocks)] += 1
    return alloc


def proportional_allocation(num_blocks, kits, current_counts=None, **kwargs):
    if num_blocks <= 0 or kits <= 0:
        return np.zeros(num_blocks, dtype=int)

    if current_counts is None:
        return uniform_allocation(num_blocks, kits)

    weights = np.zeros(num_blocks)
    for i in range(num_blocks):
        if i < len(current_counts):
            # I + Q: confirmed infectious + confirmed quarantined.
            # Q is directly observable and reflects where active cases
            # have already been detected, making it a stronger signal
            # than A (asymptomatic) which is largely unobserved.
            weights[i] = current_counts[i].get('I', 0) + current_counts[i].get('Q', 0)

    if weights.sum() == 0:
        return uniform_allocation(num_blocks, kits)

    alloc = np.floor((weights / weights.sum()) * kits).astype(int)
    remainder = kits - alloc.sum()
    if remainder > 0:
        idx = np.argsort(weights)[-remainder:]
        alloc[idx] += 1
    return alloc


# --------------------------------------------------
# Multi-Armed Bandit baseline
# --------------------------------------------------

class BetaBinomialMAB:
    """Beta-Binomial Thompson Sampling baseline."""

    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.alpha = np.ones(num_arms)
        self.beta  = np.ones(num_arms)

    def update_priors(self, history):
        if history is None:
            return
        # Reset to uniform prior then accumulate — avoids double-counting
        # across repeated calls within the same simulation run.
        self.alpha = np.ones(self.num_arms)
        self.beta  = np.ones(self.num_arms)
        for daily in history:
            for i in range(min(self.num_arms, len(daily))):
                self.alpha[i] += daily[i].get('positive', 0)
                self.beta[i]  += daily[i].get('negative', 0)

    def select_arm(self):
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))