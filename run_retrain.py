""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Retrain CNN model generated by Q-NAS.
"""

import argparse

import qnas_config as cfg
from cnn import train_detailed as train
from util import check_files, init_log


def main(**args):
    logger = init_log(args['log_level'], name=__name__)
    # Check if *experiment_path* contains all the necessary files to retrain an evolved model
    check_files(args['experiment_path'])

    # Get all parameters
    logger.info(f"Getting parameters from evolution ...")
    config = cfg.ConfigParameters(args, phase='retrain')
    config.get_parameters()

    # Load log file and read it at the specified generation
    s = f"last generation" if args['generation'] is None else f"generation {args['generation']}"
    logger.info(f"Loading {config.files_spec['data_file']} file to get {s}, individual "
                f"{args['individual']} ...")
    config.load_evolved_data(generation=args['generation'],
                             individual=args['individual'])

    if args['lr_schedule'] is not None:
        special_params = train.train_schemes_map[args['lr_schedule']].get_params()
        logger.info(f"Overriding train parameters to use special scheme "
                    f"'{args['lr_schedule']}' ...")
        config.override_train_params(special_params)

    # It is important to merge the dicts with the evolved_params first, as they need to be
    # overwritten in case we are using one of the special train schemes.
    train_params = {**config.evolved_params['params'], **config.train_spec}
    best_ind_tese = ['conv_5_1_512', 'conv_3_1_128', 'conv_3_1_512',
                     'conv_5_1_256',
                     'avg_pool_2_2',
                     'conv_3_1_256',
                     'avg_pool_2_2',
                     'conv_5_1_128',
                     'avg_pool_2_2',
                     'max_pool_2_2']

    # best_ind_tese = ['bv1p_3_1_128',
    #                  'bv1p_3_1_128',
    #                  'bv1p_3_1_256',
    #                  'avg_pool_2_2',
    #                  'no_op',
    #                  'bv1p_3_1_256',
    #                  'no_op',
    #                  'no_op',
    #                  'no_op',
    #                  'max_pool_2_2',
    #                  'max_pool_2_2',
    #                  'bv1_3_1_128',
    #                  'bv1_3_1_64',
    #                  'bv1p_3_1_256',
    #                  'no_op',
    #                  'bv1_3_1_256',
    #                  'max_pool_2_2',
    #                  'bv1_3_1_256',
    #                  'bv1p_3_1_64',
    #                  'no_op'
    #                  ]
    config.evolved_params['net'] = best_ind_tese
    logger.info(f"Starting training of model {config.evolved_params['net']}")
    valid_acc, test_info = train.train_and_eval(data_info=config.data_info,
                                                params=train_params,
                                                fn_dict=config.fn_dict,
                                                net_list=config.evolved_params['net'],
                                                lr_schedule=args['lr_schedule'],
                                                run_train_eval=args['run_train_eval'])
    logger.info(f"Saving parameters...")
    config.save_params_logfile()

    logger.info(f"Best accuracy in validation set: {valid_acc:.5f}")
    logger.info(f"Final test accuracy: {test_info['accuracy']:.5f}")
    logger.info(f"Final test confusion matrix:\n{test_info['confusion_matrix']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Directory where the evolved network logs are.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to input data.')
    parser.add_argument('--retrain_folder', type=str, default='retrain',
                        help='Name of the folder with retrain model files that will be saved '
                             'inside *experiment_path*.')
    parser.add_argument('--generation', type=int, default=None,
                        help='Generation from which evolution data will be loaded. If None, '
                             'the last generation values will be used. Default = None.')
    parser.add_argument('--individual', type=int, default=0,
                        help='Classical individual to be loaded number ID. Valid numbers: '
                             '0, ..., number of classical individuals - 1. Note that lower '
                             'numbers have higher fitness. Default = 0.')
    # Training info
    parser.add_argument('--log_level', choices=['NONE', 'INFO', 'DEBUG'], default='NONE',
                        help='Logging information level.')
    parser.add_argument('--max_epochs', type=int, default=300,
                        help='The maximum number of epochs during training. Default = 300.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Size of the batch that will be divided into multiple GPUs.'
                             ' Default = 128.')
    parser.add_argument('--eval_batch_size', type=int, default=0,
                        help='Size of the evaluation batch. Default = 0 (which means the '
                             'entire validation dataset at once).')
    parser.add_argument('--save_checkpoints_epochs', type=int, default=10,
                        help='Number of epochs to save a new checkpoint. Default = 10')
    parser.add_argument('--save_summary_epochs', type=float, default=0.25,
                        help='Number or ratio of epochs to save new info into summary. '
                             'Default = 0.25')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads to parse dataset examples and for Tensorflow '
                             'parameters *intra_op_parallelism_threads* and '
                             '*inter_op_parallelism_threads*. If = 0, the system chooses the '
                             'number equal to logical cores available in each node.')
    parser.add_argument('--lr_schedule', choices=list(train.train_schemes_map.keys()),
                        default=None,
                        help='Learning rate schedule to be used. *cosine* and *cosine500* uses '
                             'cosine decay and *special* defines epoch intervals and learning '
                             'rate values for each one. Default = None (uses evolution setup).')
    parser.add_argument('--run_train_eval', action='store_true',
                        help='If present, periodic evaluation is conducted also on the '
                             'training set.')

    arguments = parser.parse_args()

    main(**vars(arguments))
