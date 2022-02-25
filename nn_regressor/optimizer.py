# -*- coding: utf-8 -*-
import logging
import random
from copy import deepcopy
from functools import reduce
from operator import add

import numpy as np
import pandas as pd

from .model import NeuralNetworkRegressor

logger = logging.getLogger("__main__")


def optimize(
    generations,
    population,
    x_train,
    y_train,
    x_test,
    y_test,
    actual_best,
    confs_nn,
    optimization_params,
    compute_metric,
):
    """Evolve a network."""
    logger.info(
        "***Evolving %d generations with population %d***" % (generations, population)
    )

    return generate(
        generations,
        population,
        optimization_params,
        x_train,
        y_train,
        x_test,
        y_test,
        actual_best,
        confs_nn,
        compute_metric,
    )


def generate(
    generations,
    population,
    optimization_params,
    x_train,
    y_train,
    x_test,
    y_test,
    actual_best,
    confs_nn,
    compute_metric,
):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        optimization_params (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """

    optimizer = Optimizer(
        optimization_params, x_train, y_train, x_test, y_test, confs_nn, compute_metric
    )
    networks = optimizer.create_population(population, actual_best)

    # Evolve the generation.
    for i in range(generations):
        logger.info("***Doing generation %d of %d***" % (i + 1, generations))

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

        # Get the average fitness for this generation.
        average_fitness = optimizer.get_avg_fitness(networks)
        best_fitness = optimizer.get_max_fitness(networks)
        # Print out the average accuracy each generation.
        logger.info("Generation average: %.2f%%" % (average_fitness * 100))
        logger.info("Generation best: %.2f%%" % (best_fitness * 100))
        logger.info("-" * 80)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: optimizer.fitness(x), reverse=False)
    return networks


def random_create(possible_params: dict) -> dict:
    genotype = {
        "loss": random.choice(possible_params["loss"]),
        "optimizer": random.choice(possible_params["optimizer"]),
        "epochs": 10000,
        "first_layer": {
            "units": random.choice(possible_params["units"]),
            "activation": random.choice(possible_params["activation"]),
        },
        "hidden_layers": {},
    }
    hidden_layer = random.choice(possible_params["hidden_layer"])
    layer_type = None
    for layer in range(hidden_layer):
        if layer_type == "dropout":
            layer_type = "dense"  # se il layer precedente era dropout, scegli dense
        else:
            layer_type = random.choice(
                ["dropout", "dense", "dense"]
            )  # altrimenti scegli random con prob 33%-66%
        layer_name = (
            f"{layer}_{layer_type}"  #  perché pyyaml dumpa in ordine alfabetico
        )
        if layer_type == "dense":
            genotype["hidden_layers"][layer_name] = {
                "units": random.choice(possible_params["units"]),
                "activation": random.choice(possible_params["activation"]),
            }
        else:
            genotype["hidden_layers"][layer_name] = {
                "rate": random.choice(possible_params["dropout_rate"])
            }

    genotype["output_layer"] = {
        "units": 1,
        "activation": "sigmoid",
        "use_bias": random.choice([True, False]),
    }
    return genotype


class Optimizer:
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(
        self,
        possible_params,
        x_train,
        y_train,
        x_test,
        y_test,
        confs_nn,
        compute_metric,
        retain=0.4,
        random_select=0.1,
        mutate_chance=0.2,
    ):
        """Create an optimizer.

        Args:
            possible_params (dict): Possible network paremters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated

        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.possible_params = possible_params
        self.confs_nn = confs_nn

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.fitness_function = compute_metric

    def create_population(self, count, actual_best):
        """Create a population of random networks.

        Args:
            count (int): Number of networks to generate, aka the
                size of the population

        Returns:
            (list): Population of network objects

        """
        # Take as starter of the generation the current best + the standard model
        pop = [
            NeuralNetworkRegressor(actual_best),
            NeuralNetworkRegressor(self.confs_nn["default"]),
        ]
        for _ in range(1, count):
            # Create a random network and add it to our population.
            pop.append(NeuralNetworkRegressor(random_create(self.possible_params)))

        avg_fitness = self.get_avg_fitness(pop)
        max_fitness = self.get_max_fitness(pop)
        return pop

    def fitness(self, network):
        """Return the fitness function."""
        if network.metric is None:
            copied_network = deepcopy(network)
            fitness_vector = []
            n_reti = 25
            for i in range(n_reti):
                copied_network.fit(
                    self.x_train, self.y_train, verbose=0, batch_size=4096
                )
                result = pd.DataFrame(
                    {
                        "y_pred": np.squeeze(copied_network.model.predict(self.x_test)),
                        "y_true": self.y_test,
                    }
                )
                fitness_vector.append(self.fitness_function(result))
                if i != n_reti - 1:
                    copied_network = NeuralNetworkRegressor(network.parameters)
            network.metric = np.mean(fitness_vector) - np.std(fitness_vector)
            logger.info(
                f"Test set SCORE: mean={np.mean(fitness_vector)},std={np.std(fitness_vector)}, fitness={network.metric}"
            )
        logger.debug(f"Fitness: {network.metric}")
        return network.metric

    def get_max_fitness(self, pop):
        """Find min fitness for a population.

        Args:
            pop (list): The population of networks

        Returns:
            (float): The max fitness of the population

        """
        return reduce(max, (self.fitness(network) for network in pop))

    def get_avg_fitness(self, pop):
        """Find average fitness for a population.

        Args:
            pop (list): The population of networks

        Returns:
            (float): The average wmae of the population

        """
        return reduce(add, (self.fitness(network) for network in pop)) / float(
            (len(pop))
        )

    def breed(self, mother, father):
        """Make two children as parts of their parents.

        Args:
            mother (dict): Network parameters
            father (dict): Network parameters

        Returns:
            (list): Two network objects

        """
        children = []
        for _ in range(2):

            child_parameters = {}

            # Loop through the parameters and pick params for the kid.
            for param in list(mother.parameters.keys()):
                child_parameters[param] = random.choice(
                    [mother.parameters[param], father.parameters[param]]
                )

            if self.mutate_chance > random.random():
                child_parameters = self.mutate(child_parameters)

            # Now create a network object.
            network = NeuralNetworkRegressor(child_parameters)
            # Randomly mutate some of the children.
            children.append(network)

        return children

    def mutate(self, network_parameters):
        """Randomly mutate one part of the network.

        Args:
            network_parameters (dict): The network parameters to mutate

        Returns:
            (Network): A randomly mutated network object

        """
        # Choose a random key.
        random_parameters = random_create(self.possible_params)
        mutation = random.choice(list(random_parameters.keys()))

        # Mutate one of the params.
        network_parameters[mutation] = random_parameters[mutation]
        return network_parameters

    def evolve(self, pop):
        """Evolve a population of networks.

        Args:
            pop (list): A list of network parameters

        Returns:
            (list): The evolved population of networks

        """
        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=False)]

        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded) * self.retain)

        # The parents are every network we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents
