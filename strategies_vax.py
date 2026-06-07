# strategies_vax.py
# ---------------------------------------------------------------------------
# Vaccine-adapted allocation strategies.
# Mirrors the structure of strategies.py with the following changes:
#
#   1. Heuristic baselines retargeted for vaccination:
#        - uniform_allocation      : unchanged (dose-count agnostic)
#        - random_allocation       : unchanged (dose-count agnostic)
#        - proportional_allocation : targets susceptible-proxy (S estimate)
#                                    instead of I+Q
#        - risk_weighted_allocation: NEW — weights S-proxy by local infection
#                                    pressure (S_proxy * I_rate per block)
#
#   2. MAB baselines updated:
#        - BetaBinomialMABVax   : reads 'success'/'failure' instead of
#                                 'positive'/'negative'
#        - GammaPoissonMABVax   : same field rename
#
#   3. proportional_allocation uses visible I count as a proxy for
#      susceptible density.  Rationale: blocks with more active infections
#      have more unvaccinated people by definition (V nodes cannot be I).
#      A high I count is therefore a reasonable proxy for a large remaining
#      susceptible pool, especially early in the campaign.
#      Limitation: late in an epidemic a high I count may indicate the
#      susceptible pool is nearly exhausted — risk_weighted_allocation
#      partially corrects for this by discounting blocks whose I rate is
#      already declining.
#
# Public API (for simulation_vax.py):
#   uniform_allocation
#   random_allocation
#   proportional_allocation_vax
#   risk_weighted_allocation
#   BetaBinomialMABVax
#   GammaPoissonMABVax
# ---------------------------------------------------------------------------

import random
import numpy as np

from netepi_vax import S, E, I, A, V, R, STATE_NAMES_VAX


# ---------------------------------------------------------------------------
# Heuristic strategies
# ---------------------------------------------------------------------------

def uniform_allocation(num_blocks, doses_per_day, **kwargs):
    """
    Distribute doses evenly across all blocks.
    Remainder doses are assigned to the first blocks (deterministic).
    Unchanged from strategies.py — dose-count logic is allocation-agnostic.
    """
    allocations    = np.full(num_blocks, doses_per_day // num_blocks, dtype=int)
    rem_doses      = doses_per_day % num_blocks
    allocations[:rem_doses] += 1
    return allocations


def random_allocation(num_blocks, doses_per_day, **kwargs):
    """
    Assign each dose to a uniformly random block.
    Unchanged from strategies.py — dose-count logic is allocation-agnostic.
    """
    allocations = np.zeros(num_blocks, dtype=int)
    for _ in range(doses_per_day):
        allocations[random.randint(0, num_blocks - 1)] += 1
    return allocations


def proportional_allocation_vax(num_blocks, doses_per_day, current_counts,
                                 **kwargs):
    """
    Allocate doses proportional to the estimated unvaccinated population
    in each block.

    Susceptible proxy = S + E + I + A per block.
    Rationale: all of these states are unvaccinated. V and R nodes are
    excluded because vaccinating them wastes doses.

    Note: this is different from strategies.py's proportional_allocation,
    which targets I+Q (infectious detected). Here we target the full
    unvaccinated pool because the goal is protection, not detection.

    Parameters
    ----------
    current_counts : list[Counter]  — indexed by block_id,
        keys are STATE_NAMES_VAX strings ('S','E','I','A','V','R').
    """
    unvaccinated = np.array([
        counts.get('S', 0) + counts.get('E', 0)
        + counts.get('I', 0) + counts.get('A', 0)
        for counts in current_counts
    ], dtype=float)

    total_unvaccinated = unvaccinated.sum()
    if total_unvaccinated == 0:
        return uniform_allocation(num_blocks, doses_per_day)

    proportions  = unvaccinated / total_unvaccinated
    allocations  = np.floor(proportions * doses_per_day).astype(int)
    rem_doses    = doses_per_day - allocations.sum()
    if rem_doses > 0:
        residuals = (proportions * doses_per_day) - allocations
        for idx in np.argsort(residuals)[-rem_doses:]:
            allocations[idx] += 1
    return allocations


def risk_weighted_allocation(num_blocks, doses_per_day, current_counts,
                              **kwargs):
    """
    Allocate doses proportional to unvaccinated population weighted by
    local infection pressure.

    Score per block = S_proxy * I_rate
        S_proxy = S + E + I + A  (unvaccinated pool, same as above)
        I_rate  = (I + A) / block_size  (fraction actively infectious)

    Intuition: a block with many unvaccinated people AND high active
    infection is the highest-priority target — doses there both protect
    the most people and hit the hardest-hit communities.

    If all blocks have zero infection pressure (early campaign), falls
    back to proportional_allocation_vax so doses are not wasted.

    Parameters
    ----------
    current_counts : list[Counter]  — indexed by block_id.
    """
    unvaccinated = np.array([
        counts.get('S', 0) + counts.get('E', 0)
        + counts.get('I', 0) + counts.get('A', 0)
        for counts in current_counts
    ], dtype=float)

    block_sizes = np.array([
        sum(counts.values()) for counts in current_counts
    ], dtype=float)
    block_sizes = np.where(block_sizes == 0, 1, block_sizes)   # avoid /0

    infectious = np.array([
        counts.get('I', 0) + counts.get('A', 0)
        for counts in current_counts
    ], dtype=float)

    i_rate = infectious / block_sizes                           # (num_blocks,)
    scores = unvaccinated * i_rate                              # (num_blocks,)

    total_score = scores.sum()
    if total_score == 0:
        # No active infections yet — fall back to susceptible-proportional
        return proportional_allocation_vax(
            num_blocks, doses_per_day, current_counts
        )

    proportions = scores / total_score
    allocations = np.floor(proportions * doses_per_day).astype(int)
    rem_doses   = doses_per_day - allocations.sum()
    if rem_doses > 0:
        residuals = (proportions * doses_per_day) - allocations
        for idx in np.argsort(residuals)[-rem_doses:]:
            allocations[idx] += 1
    return allocations


# ---------------------------------------------------------------------------
# MAB baselines — vaccine variants
# ---------------------------------------------------------------------------

class BetaBinomialMABVax:
    """
    Thompson Sampling bandit with Beta-Binomial conjugate update.

    Vaccine adaptation of BetaBinomialMAB from strategies.py.
    Reads 'success' / 'failure' reward fields instead of
    'positive' / 'negative'.

    'success' = infection-reduction signal for that block on that day
                (computed by compute_vax_reward() in vaccination.py)
    'failure' = infections that still occurred in that block

    The posterior Beta(alpha, beta) models the probability that allocating
    a dose to block i produces a positive outcome (averts an infection).
    Thompson sampling draws from this posterior to select the block for
    each dose greedily.
    """

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.alphas     = np.ones(num_blocks)
        self.betas      = np.ones(num_blocks)

    def update_priors(self, historical_vax_results):
        """
        Reset to flat prior then accumulate all historical reward signals.

        historical_vax_results : deque of daily result lists.
            Each entry is list[dict] of length num_blocks,
            each dict has keys 'success' and 'failure'.
        """
        self.alphas = np.ones(self.num_blocks)
        self.betas  = np.ones(self.num_blocks)
        for daily_result in historical_vax_results:
            for i in range(self.num_blocks):
                self.alphas[i] += daily_result[i]['success']
                self.betas[i]  += daily_result[i]['failure']

    def select_arm(self):
        """Thompson sample: draw from each block's Beta posterior, pick max."""
        return int(np.argmax(np.random.beta(self.alphas, self.betas)))


class GammaPoissonMABVax:
    """
    Thompson Sampling bandit with Gamma-Poisson conjugate update.

    Vaccine adaptation of GammaPoissonMAB from strategies.py.
    Models the rate of infection-reduction events (successes) as Poisson,
    with a Gamma prior — appropriate when the reward signal is a count
    (number of averted infections) rather than a binary outcome.

    Reads 'success' / 'failure' reward fields.

    'success' accumulates into the Gamma shape parameter.
    The rate parameter is regularised by lookback_days so the agent
    discounts stale observations.
    """

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.shapes     = np.ones(num_blocks)
        self.rates      = np.ones(num_blocks)

    def update_priors(self, historical_vax_results, lookback_days):
        """
        Reset to flat prior then accumulate success counts.

        lookback_days controls the time-decay of the rate parameter:
        higher lookback -> lower rate -> higher variance in Thompson samples
        -> more exploration.
        """
        self.shapes = np.ones(self.num_blocks)
        self.rates  = np.ones(self.num_blocks)
        for daily_result in historical_vax_results:
            for i in range(self.num_blocks):
                self.shapes[i] += daily_result[i]['success']
        self.rates += lookback_days

    def select_arm(self):
        """Thompson sample: draw rate from Gamma posterior, pick max."""
        return int(np.argmax(np.random.gamma(self.shapes, 1.0 / self.rates)))
