# vaccination.py
# ---------------------------------------------------------------------------
# Within-block vaccination logic and lagged reward computation.
# This is the vaccine-workflow counterpart of the testing logic that lives
# inside simulation.py in the test-kit workflow.
#
# Two responsibilities:
#
#   1. node_level_vaccination()
#      Random vaccination within a single block.
#      Doses are administered to a uniformly random sample of non-I, non-V nodes.
#      Visible symptomatic I nodes are excluded (directed to isolation).
#      Already-vaccinated V nodes are excluded (no re-vaccination).
#      Outcome:
#        productive — dose given to S or E node  (S/E -> V)
#        wasted     — dose given to A or R       (no new protection conferred)
#
#   2. VaxRewardBuffer
#      Stores the per-block I-counts on the day doses are administered,
#      then releases the lagged, global-trend-corrected reward signal after
#      VACCINE_EFFECT_LAG days.
#
#      Reward formula (per block b, at lag day t + LAG):
#        global_trend  = (sum_prev_I - sum_curr_I) / num_blocks
#        raw_reduction = prev_I[b] - curr_I[b]
#        success[b]    = max(0, raw_reduction - global_trend)
#        failure[b]    = curr_I[b]
#
#      The global-trend correction prevents the agent from claiming credit
#      for natural epidemic decline that would have happened regardless of
#      vaccination.
#
# Public API (for simulation_vax.py):
#   node_level_vaccination(states, sim_graph, block_id, num_doses)
#       -> (vaccinated: int, wasted: int)
#
#   VaxRewardBuffer(num_blocks, lag_days)
#       .record_vaccination_day(day, I_counts_by_block)
#       .get_reward(day, I_counts_by_block)
#           -> list[dict{'success', 'failure'}] | None
# ---------------------------------------------------------------------------

import numpy as np
from collections import deque

from netepi_vax import S, E, I, A, V, R, Q


# ---------------------------------------------------------------------------
# Random vaccination
# ---------------------------------------------------------------------------

def node_level_vaccination(states, sim_graph, block_id, num_doses):
    """
    Administer up to num_doses vaccine doses within a single block using
    uniformly random selection.

    Candidate pool
    --------------
    All nodes in the block except visible symptomatic-I nodes and already-
    vaccinated V nodes.

    I nodes are excluded — symptomatic individuals are directed to
    isolation/treatment, not vaccination.

    V nodes are excluded — already vaccinated nodes must not receive a
    second dose. This prevents dose wastage on protected nodes and ensures
    coverage grows efficiently across the susceptible pool.
    Nodes that wane from V back to S via waning_vaccine_prob naturally
    re-enter the eligible pool at that point.

    S, E, A, R nodes are all eligible — they are indistinguishable
    without a diagnostic test under full partial observability.

    If num_doses >= len(candidates), all eligible nodes are dosed
    (no replacement sampling).

    Vaccination outcome
    -------------------
    productive : dose given to S or E  ->  node transitions to V
                 (vaccination confers future immunity)
    wasted     : dose given to A or R
                 (node already infected or naturally immune —
                  no new protection conferred)

    Parameters
    ----------
    states      : np.ndarray[int8]  — global state array, modified in-place
    sim_graph   : dict              — precomputed sim structures
    block_id    : int
    num_doses   : int

    Returns
    -------
    vaccinated  : int  — productive doses (S/E -> V)
    wasted      : int  — doses with no new protective effect
    """
    if num_doses == 0:
        return 0, 0

    block_nodes  = sim_graph['block_nodes'][block_id]
    block_states = states[block_nodes]

    # Exclude visible symptomatic I and already-vaccinated V nodes.
    # V exclusion prevents re-vaccination — once protected, a node is
    # skipped until it wanes back to S naturally.
    eligible_mask = (block_states != I) & (block_states != V)
    eligible      = block_nodes[eligible_mask]

    if len(eligible) == 0:
        return 0, 0

    # Sample without replacement; if fewer eligible nodes than doses,
    # vaccinate everyone eligible
    n_to_dose  = min(num_doses, len(eligible))
    chosen_idx = np.random.choice(len(eligible), size=n_to_dose, replace=False)
    chosen     = eligible[chosen_idx]

    vaccinated = 0
    wasted     = 0

    for node in chosen:
        st = states[node]
        if st == S or st == E:
            states[node] = V    # productive — confers immunity
            vaccinated  += 1
        else:
            # A, R, Q — dose cannot confer new protection
            wasted += 1

    return vaccinated, wasted


# ---------------------------------------------------------------------------
# Lagged, global-trend-corrected reward buffer
# ---------------------------------------------------------------------------

class VaxRewardBuffer:
    """
    Records per-block infectious counts on each vaccination day and
    releases the lagged, global-trend-corrected reward signal after
    lag_days have elapsed.

    Usage in simulation_vax.py
    --------------------------
        buffer = VaxRewardBuffer(num_blocks, config.VACCINE_EFFECT_LAG)

        # On each vaccination day:
        I_today = _count_I_by_block(states, sim_graph)
        buffer.record_vaccination_day(day, I_today)

        # After epidemic step, check if a lagged reward is ready:
        reward = buffer.get_reward(day, I_today)
        if reward is not None:
            agent.update(..., daily_vax_results=reward)
            historical_vax_results.append(reward)

    Reward formula
    --------------
    For block b:
        global_trend  = (sum(prev_I) - sum(curr_I)) / num_blocks
        raw_reduction = prev_I[b] - curr_I[b]
        success[b]    = max(0.0, raw_reduction - global_trend)
        failure[b]    = float(curr_I[b])

    global_trend correction
        Subtracting the mean per-block reduction prevents the agent from
        taking credit for epidemic decline that is happening network-wide
        (e.g. natural recovery post-peak).  Only blocks that are declining
        faster than the global average receive a positive success signal.

    failure = current I count (not the complement of success)
        This keeps the Beta distribution well-calibrated: blocks with
        ongoing high infection pressure accumulate beta mass, pulling
        future allocation towards them.  It also means success + failure
        is not constrained to a fixed total, which is correct for a
        count-valued reward rather than a binary one.
    """

    def __init__(self, num_blocks, lag_days):
        """
        Parameters
        ----------
        num_blocks : int
        lag_days   : int  — typically config.VACCINE_EFFECT_LAG (= 14)
        """
        self.num_blocks = num_blocks
        self.lag_days   = lag_days

        # Each entry: (vaccination_day, I_counts_array)
        # Stored as a deque; we look up by day index, not by position.
        self._records = deque()

    def record_vaccination_day(self, day, I_counts_by_block):
        """
        Store the per-block I counts observed on the day doses are given.
        Called once per vaccination day before the epidemic step.

        Parameters
        ----------
        day               : int
        I_counts_by_block : np.ndarray[float] shape (num_blocks,)
            Number of visible I nodes per block at time of vaccination.
        """
        self._records.append((day, np.array(I_counts_by_block, dtype=float)))

    def get_reward(self, current_day, I_counts_by_block):
        """
        Check whether the oldest stored record is now lag_days old.
        If so, compute and return the reward signal; otherwise return None.

        Parameters
        ----------
        current_day       : int
        I_counts_by_block : np.ndarray[float] shape (num_blocks,)
            Current per-block I counts (post epidemic step).

        Returns
        -------
        list[dict{'success': float, 'failure': float}] of length num_blocks,
        or None if no record has matured yet.
        """
        if not self._records:
            return None

        oldest_day, prev_I = self._records[0]

        if current_day - oldest_day < self.lag_days:
            return None

        # Consume the record
        self._records.popleft()

        curr_I = np.array(I_counts_by_block, dtype=float)

        # Global trend: average per-block change across the network
        global_trend = (prev_I.sum() - curr_I.sum()) / self.num_blocks

        reward = []
        for b in range(self.num_blocks):
            raw_reduction = float(prev_I[b] - curr_I[b])
            success = max(0.0, raw_reduction - global_trend)
            failure = float(curr_I[b])
            reward.append({'success': success, 'failure': failure})

        return reward


# ---------------------------------------------------------------------------
# Helper — count visible I nodes per block
# ---------------------------------------------------------------------------

def count_I_by_block(states, sim_graph):
    """
    Return a (num_blocks,) float array of visible symptomatic-I counts.

    Only I (symptomatic) is counted — A (asymptomatic) is not observable
    without testing, so it is excluded from the reward signal.
    This mirrors the partial-observability constraint: in the vaccine
    workflow there are no diagnostic tests to reveal A nodes.

    Parameters
    ----------
    states    : np.ndarray[int8]
    sim_graph : dict

    Returns
    -------
    np.ndarray[float] shape (num_blocks,)
    """
    num_blocks = sim_graph['num_blocks']
    I_counts   = np.zeros(num_blocks, dtype=float)
    for b in range(num_blocks):
        bn         = sim_graph['block_nodes'][b]
        I_counts[b] = float(np.sum(states[bn] == I))
    return I_counts
