# Utility for ES

import numpy as np


class SolutionState(object):
    """ Carries the current solution. Encapsulates the state-fullness of solutions during training """

    def __init__(self, initial_solution):
        """

        Args:
            initial_solution:
        """
        self._solution = initial_solution

    def set_solution(self, solution):
        """

        Args:
            solution:

        Returns:

        """
        assert solution.dtype == np.float32
        self._solution = solution

    def get_current_solution(self):
        """

        Returns:

        """
        return self._solution

    @property
    def size(self):
        """

        Returns:

        """
        return self._solution.size


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in
    [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """

    Args:
        x:

    Returns:

    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= 0.5
    return y


def itergroups(items, group_size):
    """

    Args:
        items:
        group_size:

    Returns:

    """
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def batched_weighted_sum(weights, vecs, batch_size):
    """

    Args:
        weights:
        vecs:
        batch_size:

    Returns:

    """
    total = 0
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(
            itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(
            np.asarray(batch_weights, dtype=np.float32),
            np.asarray(batch_vecs, dtype=np.float32))
        num_items_summed += len(batch_weights)
    return total, num_items_summed
