# Implements the trainer managing which manages the hole training procedure

import time
import logging
import ray
import json
import numpy as np
from ray.rllib.utils.memory import ray_get_and_free
from .noise import create_shared_noise, SharedNoiseTable
from .optimizer import Adam
from .worker import Worker
from .utils import SolutionState, compute_centered_ranks, batched_weighted_sum


logger = logging.getLogger(__name__)


class ESConfig(object):
    """ Configures the ES setup """
    def __init__(self, fitness_config, step_size=0.01, noise_size=250000000,
                 num_workers=2, evaluations_per_batch=100, l2_coefficient=0.005, noise_std_dev=0.02):
        """
        Configuration container.

        Args:
            fitness_config(`dict`): Has to be json serializable
            step_size(`float`): Learning rate of the optimizer.
            noise_size(`int`): Size of the noise table
            num_workers(`int`): Number of workers.
            evaluations_per_batch(`int`): Number of times the objective function is evaluated before calculating an
                                          update.
            l2_coefficient(float): Use default.
            noise_std_dev(`float`): Standard deviation of the noise added to the params during perturbation
        """
        self.fitness_config = fitness_config
        self.step_size = step_size
        self.noise_size = noise_size
        self.num_workers = num_workers
        self.evaluation_per_batch = evaluations_per_batch
        self.l2_coefficient = l2_coefficient
        self.noise_std_dev = noise_std_dev

    @classmethod
    def from_json(cls, path):
        """
        Create a config from json

        Args:
            path(`str`): Path to the json

        Returns:

        """
        with open(path, "r") as fj:
            config = json.load(fj)
        return ESConfig(**config)

    def to_json(self, path):
        """ Serialize config at path """
        with open(path, 'w') as fj:
            json.dump(self, fj)


class ESTrainer(object):
    """Large-scale implementation of Evolution Strategies in Ray."""

    def __init__(self, config, fitness_object_creator):
        """

        Args:
            config(ESConfig): Config container
            fitness_object_creator(`function`): Creates the fitness object. Function receives the keywords from
                                                config['fitness_config']
        """
        self.fitness = fitness_object_creator(**config.fitness_config)

        self.solution_state = SolutionState(self.fitness.initial_parameters)
        self.optimizer = Adam(self.solution_state, config.step_size)
        self.config = config
        # Create the shared noise table.
        logger.info("Creating shared noise table.")
        noise_id = create_shared_noise.remote(config.noise_size)
        self.noise = SharedNoiseTable(ray.get(noise_id))

        # Create the actors.
        logger.info("Creating actors.")
        self._workers = [
            Worker.remote(config, fitness_object_creator, noise_id)
            for _ in range(config.num_workers)
        ]

        self.episodes_so_far = 0
        self.reward_list = []
        self.t_start = time.time()

    def train_step(self):
        """

        Returns:

        """
        config = self.config

        theta = self.fitness.initial_parameters()
        assert theta.dtype == np.float32

        # Put the current policy weights in the object store.
        theta_id = ray.put(theta)
        # Use the actors to do rollouts, note that we pass in the ID of the
        # policy weights.
        results, num_episodes, num_time_steps = self._collect_results(
            theta_id=theta_id,
            min_evaluations=config.evaluation_per_batch
        )

        all_noise_indices = []
        all_training_returns = []

        # Loop over the results.
        for result in results:

            all_noise_indices += result.noise_indices
            all_training_returns += result.noisy_returns

        assert (len(all_noise_indices) == len(all_training_returns))

        self.episodes_so_far += num_episodes

        # Assemble the results.
        noise_indices = np.array(all_noise_indices)
        noisy_returns = np.array(all_training_returns)

        # Process the returns.
        proc_noisy_returns = compute_centered_ranks(noisy_returns)

        # Compute and take a step.
        g, count = batched_weighted_sum(
            proc_noisy_returns[:, 0] - proc_noisy_returns[:, 1],
            (self.noise.get(index, self.solution_state.size)
             for index in noise_indices),
            batch_size=500)
        g /= noisy_returns.size
        assert (g.shape == (self.solution_state.size, ) and g.dtype == np.float32
                and count == len(noise_indices))
        # Compute the new weights theta.
        theta, update_ratio = self.optimizer.update(-g +
                                                    config.l2_coefficient * theta)
        # Set the new weights in the local copy of the policy.
        self.solution_state.set_solution(theta)
        # Store the rewards

        info = {
            "weights_norm": np.square(theta).sum(),
            "grad_norm": np.square(g).sum(),
            "update_ratio": update_ratio,
            "episodes_this_iter": noisy_returns.size,
            "episodes_so_far": self.episodes_so_far,
        }

        reward_mean = np.mean(self.reward_list)
        result = dict(
            episode_reward_mean=reward_mean,
            info=info)

        return result

    def stop(self):
        for w in self._workers:
            w.__ray_terminate__.remote()

    def _collect_results(self, theta_id, min_evaluations):
        num_episodes, num_time_steps = 0, 0
        results = []
        while num_episodes < min_evaluations:
            logger.info(
                "Collected {} episodes so far this iter".format(num_episodes))

            rollout_ids = [
                worker.do_rollouts.remote(theta_id) for worker in self._workers
            ]
            # Get the results of the rollouts.
            for result in ray_get_and_free(rollout_ids):
                results.append(result)
                # Update the number of episodes and the number of time steps
                num_episodes += sum(len(pair) for pair in result.returns)

        return results, num_episodes, num_time_steps
