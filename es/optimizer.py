# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .utils import SolutionState


class Optimizer(object):

    """ Optimizer base class. """

    def __init__(self, solution_state):
        """
        Construct optimizer.

        Args:
            solution_state(`SolutionState`): Carries the solution state.
        """
        self.sol = solution_state
        self.dim = solution_state.size
        self.t = 0

    def update(self, globalg):
        """
        Perform Update. Requires the implementation of _compute_step.

        Args:
            globalg(`np.array`): The global gradient.

        Returns:
            new_params(`np.array`), ratio(`float`)
        """
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.sol.get_current_solution()
        ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        return theta + step, ratio

    def _compute_step(self, globalg):
        """
        Calculate a gradient

        Args:
            globalg(`np.array`): The global gradient.

        Returns:
            gradient(`np.array`)
        """
        raise NotImplementedError


class Adam(Optimizer):

    """ Implementation of Adam Optimizer """

    def __init__(self, solution_state, step_size, beta1=0.9, beta2=0.999, epsilon=1e-08):
        """
        Construct Optimizer.

        Args:
            solution_state(`SolutionState`):
            step_size(`float`): The learning rate.
            beta1(`float`): Adam Parameter. Use default value or check literature for its meaning
            beta2(`float`): Adam Parameter. Use default value or check literature for its meaning
            epsilon(`float`): Adam Parameter. Use default value or check literature for its meaning
        """
        Optimizer.__init__(self, solution_state=solution_state)
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        """
        Calculate a gradient

        Args:
            globalg(`np.array`): The global gradient.

        Returns:
            gradient(`np.array`)
        """
        a = self.step_size * (np.sqrt(1 - self.beta2 ** self.t) /
                              (1 - self.beta1**self.t))
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step
