""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Q-NAS configuration.
"""

import csv
import inspect
import os
from collections import OrderedDict
from math import sqrt
from typing import Optional, Union, get_args, get_origin

import numpy as np

from chromosome import QChromosomeNetwork
from cnn import blocks, cells, input, model
from util import load_pkl, load_yaml, natural_key


class ConfigParameters(object):
    def __init__(self, args, phase):
        """Initialize ConfigParameters.

        Args:
            args: dictionary containing the command-line arguments.
            phase: (str) one of 'evolution', 'continue_evolution' or 'retrain'.
        """

        self.phase = phase
        self.args = args
        self.QNAS_spec = {}
        # self.train_spec = {}
        self.files_spec = {}
        self.layer_dict = {}
        self.cell_list = []
        self.previous_params_file = None
        self.evolved_params = None
        self.net_list = []

    def _check_vars(self, config_file):
        """Check if all variables are in *config_file* and if their types are correct.

        Args:
            config_file: dict with parameters.
        """

        def check_cell_list():
            """Check if cell list is compatible with existing functions."""
            available_cells = [c[0] for c in inspect.getmembers(cells, inspect.isclass)]

            cell_list = config_file["QNAS"].get("cell_list")

            if cell_list:
                for cell in cell_list:
                    if cell not in available_cells:
                        raise ValueError(f"{cell} is not a valid cell!")

        def check_layer_dict():
            """Check if layer dict is compatible with existing functions."""

            available_blocks = [
                c[0] for c in inspect.getmembers(blocks, inspect.isclass)
            ]
            available_cells = [c[0] for c in inspect.getmembers(cells, inspect.isclass)]

            layer_dict = config_file["QNAS"]["layer_dict"]

            probs = []

            for name, definition in layer_dict.items():
                if "cell" in definition:
                    if definition["cell"] not in available_cells:
                        raise ValueError(f"{definition['cell']} is not a valid cell!")
                if definition["block"] not in available_blocks:
                    raise ValueError(f"{definition['block']} is not a valid block!")
                if type(definition["prob"]) == str:
                    probs.append(eval(definition["prob"]))
                else:
                    probs.append(definition["prob"])

            if any(probs):
                probs = np.sum(probs)
                if sqrt((1.0 - probs) ** 2) > 1e-2:
                    raise ValueError(
                        "Function probabilities should sum 1.0!"
                        f"But it summed to {probs}"
                        "Tolerance of numpy is 1e-2."
                    )

        vars_dict = {
            "QNAS": [
                ("crossover_rate", float),
                ("max_generations", int),
                ("max_num_nodes", int),
                ("num_quantum_ind", int),
                ("repetition", int),
                ("replace_method", str),
                ("update_quantum_rate", float),
                ("update_quantum_gen", int),
                ("save_data_freq", int),
                ("layer_dict", dict),
                ("cell_list", Optional[list]),
            ],
            "train": [
                ("batch_size", int),
                ("epochs", int),
                ("eval_epochs", int),
                ("initializations", int),
                ("folds", int),
                ("stem_filters", int),
                ("max_depth", int),
                ("dataset", str),
                ("image_size", int),
                ("num_channels", int),
                ("num_classes", int),
                ("data_augmentation", bool),
            ],
        }

        for config in vars_dict.keys():
            for item in vars_dict[config]:
                var = config_file[config].get(item[0])

                if get_origin(item[1]) == Union:
                    required_type = list(get_args(item[1]))
                else:
                    required_type = [item[1]]
                if var is None and not type(None) in required_type:
                    raise KeyError(
                        f'Variable "{config}:{item[0]}" not found in '
                        f"configuration file {self.args['config_file']}"
                    )
                elif not type(var) in required_type:
                    raise TypeError(
                        f"Variable {item[0]} should be of type {required_type} but it "
                        f"is a {type(var)}"
                    )
        check_layer_dict()
        check_cell_list()

    def _get_evolution_params(self):
        """Get specific parameters for the evolution phase."""

        config_file = load_yaml(self.args["config_file"])

        self._check_vars(
            config_file
        )  # Checking if config file contains valid information.

        self.train_spec = dict(config_file["train"])
        self.QNAS_spec = dict(config_file["QNAS"])

        self._get_layer_spec()

        self.train_spec["experiment_path"] = self.args["experiment_path"]

    def _get_layer_spec(self):
        """Organize the function specifications in *self.layer_list*, *self.layer_dict* and
        *self.QNAS_spec*.
        """

        self.QNAS_spec["layer_list"] = list(self.QNAS_spec["layer_dict"].keys())
        self.QNAS_spec["layer_list"].sort(key=natural_key)
        self.layer_dict = self.QNAS_spec["layer_dict"]
        self.cell_list = self.QNAS_spec.get("cell_list", None)

        del self.QNAS_spec["layer_dict"]

        self.QNAS_spec["initial_probs"] = []

        for layer in self.QNAS_spec["layer_list"]:
            if type(self.layer_dict[layer]["prob"]) == str:
                prob = eval(self.layer_dict[layer]["prob"])
            else:
                prob = self.layer_dict[layer]["prob"]

            # If all probabilities are None, the system assigns an equal value to all functions.
            if prob is not None:
                self.QNAS_spec["initial_probs"].append(prob)

        for item in self.layer_dict.values():
            del item["prob"]

    def _get_continue_params(self):
        """Get parameters for the continue evolution phase. The evolution parameters are loaded
        from previous evolution configuration, except from the maximum number of generations
        (*max_generations*).
        """

        self.files_spec["continue_path"] = self.args["continue_path"]
        self.files_spec["previous_QNAS_params"] = os.path.join(
            self.files_spec["continue_path"], "log_params_evolution.txt"
        )

        self.files_spec["previous_data_file"] = os.path.join(
            self.args["continue_path"], "net_list.csv"
        )
        self.load_old_params()
        self.QNAS_spec["max_generations"] = load_yaml(self.args["config_file"])["QNAS"][
            "max_generations"
        ]

        self.train_spec["experiment_path"] = self.args["experiment_path"]

    def _get_retrain_params(self):
        """Get specific parameters for the retrain phase. The keys in *self.train_spec* that
        exist in self.args are overwritten.
        """

        self.files_spec["previous_QNAS_params"] = os.path.join(
            self.args["experiment_path"], "log_params_evolution.txt"
        )
        self.load_old_params()

        for key in self.args.keys():
            self.train_spec[key] = self.args[key]

        with open(os.path.join(self.train_spec["experiment_path"], self.args["id_num"], 'net_list.csv'), newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                self.net_list = row

        self.train_spec["experiment_path"] = os.path.join(
            self.train_spec["experiment_path"], self.args["retrain_folder"]
        )
        del self.args["retrain_folder"]


    def _get_common_params(self):
        """Get parameters that are combined/calculated the same way for all phases."""

        self.train_spec["data_path"] = self.args["data_path"]

        self.train_spec["phase"] = self.phase
        self.train_spec["log_level"] = self.args["log_level"]

        self.files_spec["log_file"] = os.path.join(
            self.args["experiment_path"], "log_QNAS.txt"
        )
        self.files_spec["data_file"] = os.path.join(
            self.args["experiment_path"], "net_list.csv"
        )

    def get_parameters(self):
        """Organize dicts combining the command-line and config_file parameters,
        joining all the necessary information for each *phase* of the program.
        """

        if self.phase == "evolution":
            self._get_evolution_params()
        elif self.phase == "continue_evolution":
            self._get_continue_params()
        else:
            self._get_retrain_params()

        self._get_common_params()

    def load_old_params(self):
        """Load parameters from *self.files_spec['previous_QNAS_params']* and replace
        *self.train_spec*, *self.QNAS_spec*, and *self.layer_dict* with the file values.
        """

        previous_params_file = load_yaml(self.files_spec["previous_QNAS_params"])

        self.train_spec = dict(previous_params_file["train"])
        self.QNAS_spec = dict(previous_params_file["QNAS"])
        #self.QNAS_spec["params_ranges"] = eval(self.QNAS_spec["params_ranges"])
        self.layer_dict = previous_params_file["layer_dict"]
        self.cell_list = previous_params_file["cell_list"]

    def load_evolved_data(self, generation=None, individual=0):
        """Read the yaml log *self.files_spec['data_file']* and get values from the individual
            specified by *generation* and *individual*.

        Args:
            generation: (int) generation number from which data will be loaded. If None, loads
                the last generation data.
            individual: (int) number of the classical individual to be loaded. If no number is
                specified, individual 0 is loaded (the one with highest fitness on the given
                *generation*.
        """


        with open(self.files_spec["data_file"], newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                net_list = row

        self.evolved_params = {"net": net}

    def override_train_params(self, new_params_dict):
        """Override *self.train_spec* parameters with the ones in *new_params_dict*. Update
            step parameters, in case a epoch parameter was modified.

        Args:
            new_params_dict: dict containing parameters to override/add to self.train_spec.
        """

        self.train_spec.update(new_params_dict)

    def params_to_logfile(self, params, text_file, nested_level=0):
        """Print dictionary *params* to a txt file with nested level formatting.

        Args:
            params: dictionary with parameters.
            text_file: file object.
            nested_level: level of nested dictionary.
        """

        spacing = "    "
        if type(params) == dict:
            for key, value in OrderedDict(sorted(params.items())).items():
                if type(value) == dict:
                    if nested_level < 2:
                        print(f"{nested_level * spacing}{key}:", file=text_file)
                        self.params_to_logfile(value, text_file, nested_level + 1)
                    else:
                        print(f"{nested_level * spacing}{key}: {value}", file=text_file)
                else:
                    if type(value) == float:
                        if value < 1e-3:
                            print(
                                f"{nested_level * spacing}{key}: {value:.2E}",
                                file=text_file,
                            )
                        else:
                            print(
                                f"{nested_level * spacing}{key}: {value:.4f}",
                                file=text_file,
                            )
                    else:
                        print(f"{nested_level * spacing}{key}: {value}", file=text_file)
                if nested_level == 0:
                    print("", file=text_file)

    def save_params_logfile(self):
        """Helper function to save the parameters in a txt file."""
        if self.train_spec["phase"] == "retrain":
            phase = "retrain"
            params_dict = {
                "evolved_params": self.evolved_params,
                "train": self.train_spec,
                "files": self.files_spec,
            }
        else:
            phase = "evolution"
            params_dict = {
                "QNAS": self.QNAS_spec,
                "train": self.train_spec,
                "files": self.files_spec,
                "layer_dict": self.layer_dict,
                "cell_list": self.cell_list,
            }

        params_file_path = os.path.join(
            self.train_spec["experiment_path"], f"log_params_{phase}.txt"
        )

        with open(params_file_path, mode="w") as text_file:
            self.params_to_logfile(params_dict, text_file)
