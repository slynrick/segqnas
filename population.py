""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Quantum population classes.
"""

import numpy as np

from chromosome import QChromosomeNetwork


class QPopulation(object):
    """QNAS Population to be evolved."""

    def __init__(self, num_quantum_ind, repetition, update_quantum_rate):
        """Initialize QPopulation.

        Args:
            num_quantum_ind: (int) number of quantum individuals.
            repetition: (int) ratio between the number of classic individuals in the classic
                population and the quantum individuals in the quantum population.
            update_quantum_rate: (float) probability that a quantum gene will be updated.
        """

        self.dtype = np.float64  # Type of quantum population arrays.

        self.chromosome = None
        self.current_pop = None
        self.num_ind = num_quantum_ind

        self.repetition = repetition
        self.update_quantum_rate = update_quantum_rate

    def initialize_qpop(self):
        raise NotImplementedError(
            "initialize_qpop() must be implemented in sub classes"
        )

    def generate_classical(self):
        raise NotImplementedError(
            "generate_classical() must be implemented in sub classes"
        )

    def update_quantum(self, intensity):
        raise NotImplementedError("update_quantum() must be implemented in sub classes")


class QPopulationNetwork(QPopulation):
    """QNAS Chromosomes for the networks to be evolved."""

    def __init__(
        self,
        num_quantum_ind,
        max_num_nodes,
        repetition,
        update_quantum_rate,
        layer_list,
        initial_probs,
    ):
        """Initialize QPopulationNetwork.

        Args:
            num_quantum_ind: (int) number of quantum individuals.
            max_num_nodes: (int) maximum number of nodes of the network, which will be the
                number of genes in a individual.
            repetition: (int) ratio between the number of classic individuals in the classic
                population and the quantum individuals in the quantum population.
            update_quantum_rate: (float) probability that a quantum gene will be updated.
            layer_list: list of possible functions.
            initial_probs: list defining the initial probabilities for each function; if empty,
                the algorithm will give the same probability for each function.
        """

        super(QPopulationNetwork, self).__init__(
            num_quantum_ind, repetition, update_quantum_rate
        )
        self.probabilities = None

        self.max_update = 0.05
        self.max_prob = 0.99

        self.chromosome = QChromosomeNetwork(max_num_nodes, layer_list, self.dtype)

        self.initial_probs = self.chromosome.initialize_qgenes(
            initial_probs=initial_probs
        )
        self.initialize_qpop()

    def initialize_qpop(self):
        """Initialize quantum population with *self.num_ind* individuals."""

        # Shape = (num_ind, num_nodes, num_functions)
        self.probabilities = np.tile(
            self.initial_probs, (self.num_ind, self.chromosome.num_genes, 1)
        )

    def generate_classical(self):
        """Generate a specific number of classical individuals from the observation of quantum
        individuals. This number is equal to (*num_ind* x *repetition*).
        """

        def sample(idx0, idx1):
            return np.random.choice(size, p=temp_prob[idx0, idx1, :])

        size = self.chromosome.num_functions
        new_pop = np.zeros(
            shape=(self.num_ind * self.repetition, self.chromosome.num_genes),
            dtype=np.int32,
        )

        temp_prob = np.tile(self.probabilities, (self.repetition, 1, 1))

        for ind in range(self.num_ind * self.repetition):
            for node in range(self.chromosome.num_genes):
                new_pop[ind, node] = sample(ind, node)

        return new_pop

    def _update(self, chromosomes, idx, update_value):
        """Modify *chromosomes* by adding *update_value* to the genes indicated by *idx* and
            subtracting *update_value* from the other genes proportional to the size of each
            probability.

        Args:
            chromosomes: 2D float numpy array representing the chromosomes to be updated.
            idx: (int) index of the genes to have their value increased.
            update_value: (float) value that will be added to the selected functions in
                *chromosomes* by *idx*.

        Returns:
            modified chromosome
        """

        idx0 = np.arange(chromosomes.shape[0])
        update_array = np.where(
            chromosomes[idx0, idx] + update_value > self.max_prob, 0, update_value
        )
        sum_values = chromosomes[idx0, idx] + update_array
        chromosomes[idx0, idx] = 0
        decrease = (update_array / np.sum(chromosomes, axis=1)).reshape(-1, 1)
        decrease = decrease * chromosomes
        chromosomes = chromosomes - decrease
        chromosomes[idx0, idx] = sum_values

        return chromosomes

    def update_quantum(self, intensity):
        """Update self.probabilities.

        Args:
            intensity: (float) value defining the intensity of the update.
        """

        random = np.random.rand(self.num_ind, self.chromosome.num_genes)
        mask = np.where(random <= self.update_quantum_rate)

        update_value = intensity * self.max_update

        best_classic = self.current_pop[: self.num_ind]
        self.probabilities[mask] = self._update(
            self.probabilities[mask], best_classic[mask], update_value
        )
