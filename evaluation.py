""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Distribute population eval using MPI.
"""

import time
from multiprocessing import Process, Value

import numpy as np
from cnn import train
from util import init_log


class EvalPopulation(object):
    def __init__(self, train_params, layer_dict, cell_list=None, log_level="INFO"):
        """Initialize EvalPopulation.

        Args:
            train_params: dictionary with parameters.
            layer_dict: dict with definitions of the functions (name and parameters);
                format --> {'layer_name': ['layerClass', {'param1': value1, 'param2': value2}]}.
            cell_list: list of predefined cell types that define a topology;
                format --> e.g. ['DownscalingCell', 'NonscalingCell', 'UpscalingCell', ...]
            log_level: (str) one of "INFO", "DEBUG" or "NONE".
        """

        self.train_params = train_params
        self.layer_dict = layer_dict
        self.cell_list = cell_list
        self.logger = init_log(log_level, name=__name__)

    def __call__(self, decoded_nets, generation, arch_memory):
        """Train and evaluate *decoded_nets*

        Args:
            decoded_nets: list containing the lists of network layers descriptions
                (size = num_individuals).
            generation: (int) generation number.

        Returns:
            numpy array containing evaluations results of each model in *net_list*.
        """

        pop_size = len(decoded_nets)

        evaluations = np.empty(shape=(pop_size,))
        print(self.layer_dict)

        variables = [Value('f', 0.0) for _ in range(pop_size)]
        selected_thread = 0
        individual_per_thread = []
        for idx in range(len(variables)):
            self.logger.info(f"Going to start fitness of individual {idx} on thread {selected_thread}")
            fitness = None
            key = '+'.join(decoded_nets[idx])
            if key in arch_memory:
                fitness = arch_memory[key]['fitness']
            individual_per_thread.append((idx, selected_thread, decoded_nets[idx], variables[idx], fitness))
            selected_thread += 1
            if selected_thread >= self.train_params['threads']:
                selected_thread = selected_thread % self.train_params['threads']
            
            
        processes = []
        for idx in range(self.train_params['threads']):
            individuals_selected_thread = list(filter(lambda x: x[1]==idx, individual_per_thread))
            print(individuals_selected_thread)
            process = Process(target=self.run_individuals, args=(generation, individuals_selected_thread))
            process.start()
            processes.append(process)

        for p in processes:
            p.join()
                    
        for idx, val in enumerate(variables):
            evaluations[idx] = val.value
        

        return evaluations
    
    def run_individuals(self, generation, individuals_selected_thread):
        for individual, selected_gpu, decoded_net, return_val, fitness in individuals_selected_thread:
            print(f"starting individual {individual}")
            if fitness is None:
                train.fitness_calculation(
                    id_num=f"{generation}_{individual}",
                    train_params={**self.train_params},
                    layer_dict=self.layer_dict,
                    net_list=decoded_net,
                    cell_list=self.cell_list,
                    return_val=return_val
                )
            else:
                return_val.value = fitness
                print("cached individual")
            print(f"finishing individual {individual} - {return_val.value}")
            self.logger.info(f"Clculated fitness of individual {individual} on thread {selected_gpu} with {return_val.value}")

