# Implements the trainer managing which manages the hole training procedure

import time
import ray
import json
import numpy as np
import os
import h5py
from ray.rllib.utils.memory import ray_get_and_free
from .noise import create_shared_noise, SharedNoiseTable
from .optimizer import Adam
from .worker import Worker
from .utils import SolutionState, compute_centered_ranks, batched_weighted_sum


class ESConfig(object):
    """ Configures the ES setup """
    def __init__(self, fitness_config, training_steps=100, step_size=0.01, noise_size=250000000,
                 num_workers=2, evaluations_per_batch=100, l2_coefficient=0.005, noise_std_dev=0.02,
                 experiment_dir=None, save_t=0, seed=1):
        """
        Configuration container.

        Args:
            fitness_config(`dict`): Has to be json serializable. These parameters will be forwarded to the
                                    create_fitness_object function passed to the ESTrainer.
            training_steps(`int`): Number of training steps to perform
            step_size(`float`): Learning rate of the optimizer.
            noise_size(`int`): Size of the noise table
            num_workers(`int`): Number of workers.
            evaluations_per_batch(`int`): Number of times the objective function is evaluated before calculating an
                                          update.
            l2_coefficient(float): Use default.
            noise_std_dev(`float`): Standard deviation of the noise added to the params during perturbation
            experiment_dir(`str`): Path to a dir where training statistics will be stored. The dir must not exist yet.
                                   If None, nothing will be saved
            save_t(`int`): number of training steps between saving the solution. if 0, solutions are never saved.
                           Note, that the experiment_dir parameter must not be None in order to save.
            seed(`int`): Seed for the noise table sampling.
        """
        self.fitness_config = fitness_config
        self.training_steps = training_steps
        self.step_size = step_size
        self.noise_size = noise_size
        self.num_workers = num_workers
        self.evaluation_per_batch = evaluations_per_batch
        self.l2_coefficient = l2_coefficient
        self.noise_std_dev = noise_std_dev
        self.experiment_dir = experiment_dir
        self.save_t = save_t
        self.seed = seed

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
            json.dump(self.__dict__, fj)


class ESTrainer(object):
    """Large-scale implementation of Evolution Strategies in Ray."""

    def __init__(self, config, fitness_object_creator):
        """
        Construct Object.

        Args:
            config(ESConfig): Config container.
            fitness_object_creator(`function`): Creates the fitness object. Function receives the keywords from
                                                config['fitness_config']
        """
        self.fitness = fitness_object_creator(**config.fitness_config)

        # A container that helps us keep track of the current solution
        # Here, its initialized
        self.solution_state = SolutionState(self.fitness.get_initial_solution())

        # create Adam optimizer
        self.optimizer = Adam(self.solution_state, config.step_size)

        # save the config
        self.config = config

        # Create the shared noise table.
        print("Creating shared noise table.")
        self.noise_id = create_shared_noise.remote(config.noise_size)
        self.noise = SharedNoiseTable(ray.get(self.noise_id))

        # Create the actors.
        print("Creating actors.")
        rng = np.random.RandomState(config.seed)
        worker_seeds = [rng.randint(0, 100000000) for _ in range(config.num_workers)]
        self._workers = [
            Worker.remote(worker_seeds[i], config, fitness_object_creator, self.noise_id, self.solution_state.size)
            for i in range(config.num_workers)
        ]

        self.episodes_so_far = 0
        self.reward_list = []
        self.t_start = time.time()

        if self.config.experiment_dir is not None:
            assert not os.path.isdir(config.experiment_dir), "Folder already exists!"
            os.mkdir(config.experiment_dir)
            self.statistic_file = h5py.File(os.path.join(self.config.experiment_dir, "training_log.h5"), 'w')
            self.statistic_file.create_dataset("weights_norm", dtype=np.float32, shape=[self.config.training_steps])
            self.statistic_file.create_dataset("grad_norm", dtype=np.float32, shape=[self.config.training_steps])
            self.statistic_file.create_dataset("update_ratio", dtype=np.float32, shape=[self.config.training_steps])
            self.statistic_file.create_dataset("episodes_this_iter", dtype=np.int32, shape=[self.config.training_steps])
            self.statistic_file.create_dataset("episodes_so_far", dtype=np.int32, shape=[self.config.training_steps])
            self.statistic_file.create_dataset(
                "episode_reward_mean", dtype=np.float32, shape=[self.config.training_steps]
            )
            self.statistic_file.close()

    def _store_result(self, step, result):
        """
        Stores one result in the h5 file

        Args:
            step(`int`): Update step. Defines location of this result in the array from the h5 file.
            result(`dict`): Single step result

        Returns:
            None
        """
        file = h5py.File(os.path.join(self.config.experiment_dir, "training_log.h5"), 'a')
        for key, value in result.items():
            file[key][step] = value

    def _store_solution(self, step):
        """
        Store current solution

        Args:
            step(`int`): Current update step. Visible in file name
        """
        with h5py.File(os.path.join(self.config.experiment_dir, "solution_{}.h5").format(step), 'w') as fh:
            fh['solution'] = self.solution_state.get_current_solution()

    def train_step(self):
        """
        Perform one training step. This means one parameter update.
            - put the current solution in the object store
            - run evaluations on the workers
            - collect their results
            - calculate parameter update
            - update parameters and set new solution (solution = new params)
            - return results for statistic collection

        Returns:
            results(`dict`): Information about the training step.
        """
        config = self.config

        # get the current solution
        theta = self.solution_state.get_current_solution()
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
            all_training_returns += result.returns

        assert (len(all_noise_indices) == len(all_training_returns))

        self.episodes_so_far += num_episodes

        # Assemble the results.
        noise_indices = np.array(all_noise_indices)
        returns = np.array(all_training_returns)

        # Process the returns.
        proc_noisy_returns = compute_centered_ranks(returns)

        # Compute and take a step.
        g, count = batched_weighted_sum(
            proc_noisy_returns[:, 0] - proc_noisy_returns[:, 1],
            (self.noise.get(index, self.solution_state.size)
             for index in noise_indices),
            batch_size=500)
        g /= returns.size
        assert (g.shape == (self.solution_state.size, ) and g.dtype == np.float32
                and count == len(noise_indices))
        # Compute the new weights theta.
        theta, update_ratio = self.optimizer.update(-g +
                                                    config.l2_coefficient * theta)
        # Set the new weights in the local copy of the policy.
        self.solution_state.set_solution(theta)
        # Store the rewards
        self.reward_list += list(returns)
        reward_mean = np.mean(returns)
        info = {
            "weights_norm": np.square(theta).sum(),
            "grad_norm": np.square(g).sum(),
            "update_ratio": update_ratio,
            "episodes_this_iter": returns.size,
            "episodes_so_far": self.episodes_so_far,
            "episode_reward_mean": reward_mean
        }

        return info

    def stop(self):
        """ Terminate all workers. """
        for w in self._workers:
            w.__ray_terminate__.remote()

    def _collect_results(self, theta_id, min_evaluations):
        """
        Calls the worker and tell them to run evaluations.
        The parameter `min_evaluations` controls the minimal amount of evaluation to take.
        It should be noted though that there is also a minimal time limit of 200 ms which means if evaluations
        are really fast there can be far more than min_evaluations.

        Args:
            theta_id(`ObjectId`): Object id from the ray object store
            min_evaluations(`int`): Min evals to run

        Returns:
            results(`list`): List of namedtuple containing calculations from the workers

        """
        num_episodes, num_time_steps = 0, 0
        results = []
        while num_episodes < min_evaluations:

            rollout_ids = [
                worker.do_rollouts.remote(theta_id) for worker in self._workers
            ]
            # Get the results of the rollouts.
            for result in ray_get_and_free(rollout_ids):
                results.append(result)
                # Update the number of episodes and the number of time steps
                num_episodes += sum(len(pair) for pair in result.returns)

        return results, num_episodes, num_time_steps

    def train(self):
        """
        Run the train loop. Each step contains at least `evaluations_per_step` evaluations but it can be more
        since each worker gets at least 200 ms to run objective function evaluations.

        Returns:
            results(`list`): List of dicts containing information about every training step

        """
        results = []

        def make_key_nice(k):
            return " ".join([w.capitalize() for w in k.split('_')])

        for t in range(self.config.training_steps):
            result = self.train_step()
            if t % 1 == 0:
                print("********** Iteration {} ***********".format(t))
                for key, value in result.items():
                    print("\t{}: {}".format(make_key_nice(key), value))
            results.append(result)
            if self.config.experiment_dir is not None:
                self._store_result(t, result)
                if self.config.save_t > 0 and t % self.config.save_t == 0:
                    self._store_solution(t)

        return results
