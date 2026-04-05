# strategies.py (UPDATED)
# Contains ONLY non-GNTS strategies (clean separation)

import numpy as np

# --------------------------------------------------
# Heuristic strategies
# --------------------------------------------------

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
            # Use I + Q (observable signal)
            weights[i] = current_counts[i].get('I', 0) + current_counts[i].get('Q', 0)

    if weights.sum() == 0:
        return uniform_allocation(num_blocks, kits)

    proportions = weights / weights.sum()

    allocations = np.floor(proportions * kits).astype(int)
    remainder = kits - allocations.sum()

    if remainder > 0:
        residuals = (proportions * kits) - allocations
        allocations[np.argsort(residuals)[-remainder:]] += 1

    return allocations


# --------------------------------------------------
# Beta-Binomial Multi-Armed Bandit
# --------------------------------------------------

class BetaBinomialMAB:
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.alphas = np.ones(num_blocks)
        self.betas = np.ones(num_blocks)

    def update_priors(self, historical_test_results):
        self.alphas = np.ones(self.num_blocks)
        self.betas = np.ones(self.num_blocks)

        for daily_result in historical_test_results:
            for i in range(self.num_blocks):
                self.alphas[i] += daily_result[i]['positive']
                self.betas[i] += daily_result[i]['negative']

    def select_arm(self):
        samples = np.random.beta(self.alphas, self.betas)
        return np.argmax(samples)
