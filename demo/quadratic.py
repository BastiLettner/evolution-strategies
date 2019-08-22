# Optimize a quadratic function. (no distribution but two worker single machine)

import ray
import numpy as np
from es.trainer import ESConfig, ESTrainer
from es.fitness import Fitness
from es.utils import plot_training_log


class Quadratic(Fitness):
    """
    y = -(x - 5)**2 -> x_hat = 5 is max
    """
    def evaluate(self, params):
        return float(-(params[0] - 5)**2)

    def get_initial_solution(self):
        return np.array([0.0], dtype=np.float32)


def main():

    ray.init(num_cpus=2, object_store_memory=int(1e8))

    config = ESConfig(
        fitness_config={},
        noise_size=int(1e7),
        num_workers=2,
        step_size=0.5,
        training_steps=50
    )

    trainer = ESTrainer(config=config, fitness_object_creator=lambda : Quadratic())
    trainer.train()

    print("Final Solution {}".format(trainer.solution_state.get_current_solution()[0]))


if __name__ == '__main__':
    main()
