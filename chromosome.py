""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Quantum chromosomes classes.
"""

import numpy as np


class QChromosome(object):
    """QNAS Chromosomes to be evolved."""

    def __init__(self, dtype):
        """Initialize QChromosome.

        Args:
            dtype: type of the chromosome array.
        """

        self.num_genes = None
        self.dtype = dtype

    def initialize_qgenes(self, *args):
        raise NotImplementedError(
            "initialize_qgenes() must be implemented in sub classes"
        )

    def set_num_genes(self, num_genes):
        """Set the number of genes of the chromosome.

        Args:
            num_genes: (int) number of genes.
        """

        self.num_genes = num_genes

    def decode(self, chromosome):
        raise NotImplementedError("decode() must be implemented in sub classes")


class QChromosomeNetwork(QChromosome):
    def __init__(self, max_num_nodes, layer_list, dtype=np.float64):
        """Initialize QChromosomeNetwork.

        Args:
            max_num_nodes: (int) maximum number of nodes of the network, which will be the
                number of genes.
            layer_list: list of possible functions.
            dtype: type of the chromosome array.
        """

        super(QChromosomeNetwork, self).__init__(dtype)

        self.layer_list = layer_list
        self.num_functions = len(self.layer_list)

        self.set_num_genes(max_num_nodes)

    def initialize_qgenes(self, initial_probs=None):
        """Get the initial values for probabilities based on the available number of
            functions of a node if *initial_probs* is empty.

        Args:
            initial_probs: list defining the initial probabilities for each function.

        Returns:
            initial probabilities for quantum individual.
        """

        if not initial_probs:
            prob = 1 / self.num_functions
            initial_probs = np.full(
                shape=(self.num_functions,), fill_value=prob, dtype=self.dtype
            )
        else:
            initial_probs = np.array(initial_probs)

        return initial_probs

    def decode(self, chromosome):
        """Convert numpy array representing the classic chromosome into a list of function
            names representing the layers of the network.

        Args:
            chromosome: int numpy array, containing indexes that will be used to get the
                corresponding function names in self.layer_list.

        Returns:
            list with function names, in the order they represent the network.
        """

        decoded = [None] * chromosome.shape[0]

        for i, gene in enumerate(chromosome):
            if gene >= 0:
                decoded[i] = self.layer_list[gene]

        return decoded
