# Implementation of the ES worker using Ray
# Very close to https://github.com/ray-project/ray/blob/master/rllib/agents/es/es.py

import time
import numpy as np
import ray
from collections import namedtuple
from .noise import SharedNoiseTable


Result = namedtuple("Result", ["noise_indices", "returns", "sign_noisy_returns"])


@ray.remote
class Worker(object):

    """ ES Worker. Executed (potentially) distributed with the help of Ray """

    def __init__(self,
                 config,
                 fitness_object_creator,
                 noise,
                 solution_size,
                 min_task_runtime=0.2):
        """

        Args:
            config(`ESConfig`): Config container
            fitness_object_creator(`function`): Creates the fitness object
            noise(`np.array`): The noise array (shared between workers)
            solution_size(`int`): Size of each solution
            min_task_runtime(`float`): Min runtime for a rollout in seconds
        """
        self.min_task_runtime = min_task_runtime
        self.config = config
        self.solution_size = solution_size
        self.noise = SharedNoiseTable(noise)
        self.fitness = fitness_object_creator(**config.fitness_config)

    def do_rollouts(self, params):
        """
        Perform evaluations using the provided params.

        Args:
            params(`np.array`): The parameters this worker should use for evaluation

        Returns:
            Result(`named_tuple`)
        """
        noise_indices, returns, sign_returns = [], [], []

        # Perform some rollouts with noise.
        task_t_start = time.time()
        while len(noise_indices) == 0 or time.time() - task_t_start < self.min_task_runtime:

                noise_index = self.noise.sample_index(dim=self.solution_size)

                perturbation = self.config.noise_std_dev * self.noise.get(noise_index, self.solution_size)

                rewards_pos = self.fitness.evaluate(params + perturbation)

                rewards_neg = self.fitness.evaluate(params - perturbation)

                noise_indices.append(noise_index)
                returns.append([rewards_pos, rewards_neg])
                sign_returns.append(
                    [np.sign(rewards_pos),
                     np.sign(rewards_neg)])

        return Result(
            noise_indices=noise_indices,
            returns=returns,
            sign_noisy_returns=sign_returns
        )
