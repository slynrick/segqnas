""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Retrain CNN model generated by Q-NAS.
"""

import argparse

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

import qnas_config as cfg
from cnn import train_detailed
from util import init_log


def main(**args):
    logger = init_log(args["log_level"], name=__name__)

    # Get all parameters
    logger.info("Getting parameters from evolution ...")
    config = cfg.ConfigParameters(args, phase="retrain")
    config.get_parameters()
    
    config.train_spec['batch_size'] = args['batch_size']
    config.train_spec['epochs'] = args['max_epochs']
    config.train_spec['eval_epochs'] = args['eval_epochs']
    config.train_spec['initializations'] = args['initializations']
    
    gen, ind = args["id_num"].split("_")
    gen = int(gen)
    ind = int(ind)

    config.load_evolved_data(gen, ind)

    logger.info("Starting training of model")
    if config.cell_list == 'None':
        config.cell_list = None
    mean_dsc, std_dsc, test_dice = train_detailed.fitness_calculation(
        args["id_num"], config.train_spec, config.layer_dict, config.evolved_params['net'], config.cell_list
    )

    logger.info("Saving parameters...")
    config.save_params_logfile()

    logger.info(f"Final Val: {mean_dsc} +- {std_dsc}")
    logger.info(f"Final Test: {test_dice}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_path", type=str, required=True, help="Path to experiment_path."
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to input data."
    )
    parser.add_argument(
        "--retrain_folder",
        type=str,
        required=True,
        help="Path where the retrained results will be saved.",
    )
    parser.add_argument(
        "--id_num",
        type=str,
        required=True,
        help="id_num of the individual to be retrained.",
    )
    # Training info
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Size of the batch that will be divided into multiple GPUs."
        " Default = 128.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="The maximum number of epochs during training. Default = 100.",
    )
    parser.add_argument(
        "--eval_epochs",
        type=int,
        default=10,
        help="The number of epochs at the end of the training used to average the dice coeficient. Default = 10.",
    )
    parser.add_argument(
        "--initializations",
        type=int,
        default=5,
        help="The number of initializations for the cross validation. Default = 5.",
    )
    parser.add_argument(
        "--log_level",
        choices=["NONE", "INFO", "DEBUG"],
        default="NONE",
        help="Logging information level.",
    )

    arguments = parser.parse_args()

    main(**vars(arguments))