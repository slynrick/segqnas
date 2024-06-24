""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Run Q-NAS evolution.
"""
import argparse
import os

import evaluation as evaluation
import qnas
import qnas_config as cfg
from util import check_files, init_log

from multiprocessing import set_start_method
set_start_method('spawn', True)

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

def run(**args):
    logger = init_log(args['log_level'], name=__name__)

    if not os.path.exists(args['experiment_path']):
        logger.info(f"Creating {args['experiment_path']} ...")
        os.makedirs(args['experiment_path'])

    # Evolution or continue previous evolution
    if not args['continue_path']:
        phase = 'evolution'
    else:
        phase = 'continue_evolution'
        logger.info(f"Continue evolution from: {args['continue_path']}. Checking files ...")
        check_files(args['continue_path'])

    logger.info(f"Getting parameters from {args['config_file']} ...")
    config = cfg.ConfigParameters(args, phase=phase)
    config.get_parameters()
    if config.cell_list == 'None':
        config.cell_list = None
    logger.info(f"Saving parameters for {config.phase} phase ...")
    config.save_params_logfile()

    # Evaluation function for QNAS (train CNN and return validation accuracy)
    eval_f = evaluation.EvalPopulation(
        train_params=config.train_spec,
        layer_dict=config.layer_dict,
        cell_list=config.cell_list,
        log_level=config.train_spec["log_level"],
    )

    qnas_cnn = qnas.QNAS(
        eval_f, config.train_spec['experiment_path'],
        log_file=config.files_spec['log_file'],
        log_level=config.train_spec['log_level'],
        data_file=config.files_spec['data_file']
    )

    qnas_cnn.initialize_qnas(**config.QNAS_spec)

    # If continue previous evolution, load log file and read it at final generation
    if phase == 'continue_evolution':
        logger.info(f"Loading {config.files_spec['previous_data_file']} file to get final "
                    f"generation ...")
        qnas_cnn.load_qnas_data(file_path=config.files_spec['previous_data_file'])

    # Execute evolution
    logger.info(f"Starting evolution ...")
    qnas_cnn.evolve()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Directory where to write logs and model files.')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Configuration file name.')
    parser.add_argument('--continue_path', type=str, default='',
                        help='If the user wants to continue a previous evolution, point to '
                             'the corresponding experiment path. Evolution parameters will be '
                             'loaded from this folder.')
    parser.add_argument('--log_level', choices=['NONE', 'INFO', 'DEBUG'], default='NONE',
                        help='Logging information level.')

    arguments = parser.parse_args()

    run(**vars(arguments))
