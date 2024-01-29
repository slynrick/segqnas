""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Retrain CNN model generated by Q-NAS.
"""

import argparse
from multiprocessing import Value

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

import qnas_config as cfg
from cnn import train
from util import init_log


def main(**args):
    logger = init_log(args["log_level"], name=__name__)

    # Get all parameters
    logger.info(f"Getting parameters from evolution ...")
    config = cfg.ConfigParameters(args, phase="retrain")
    config.get_parameters()
    
    config.train_spec['batch_size'] = args['batch_size']
    config.train_spec['epochs'] = args['max_epochs']
    config.train_spec['eval_epochs'] = args['eval_epochs']
    config.train_spec['initializations'] = args['initializations']
    config.train_spec['folds'] = args['folds']
    
    gen, ind = args["id_num"].split("_")
    gen = int(gen)
    ind = int(ind)

    config.load_evolved_data(gen, ind)
    # print(config.args)
    # print(config.QNAS_spec)
    # print(config.files_spec)
    # print(config.layer_dict)
    # print(config.cell_list)
    # print(config.previous_params_file)
    # print(config.evolved_params['net'])

    # It is important to merge the dicts with the evolved_params first, as they need to be
    # overwritten in case we are using one of the special train schemes.

    logger.info(f"Starting training of model")
    results = Value('f', 0.0)
    train.fitness_calculation(
        args["id_num"], config.train_spec, config.layer_dict, config.evolved_params['net'], results, config.cell_list
    )

    # print(args)
    # print(id_num)
    # print(train_params)
    # print(layer_dict)

    # with open(os.path.join(args['experiment_path'], args['id_num'], 'net_list.csv'), newline='') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         net_list = row

    # logger.info(f"Starting training of model {config.evolved_params['net']}")
    # valid_mean_iou, test_info = train.train_and_eval(
    #     data_info=config.data_info,
    #     params=train_params,
    #     layer_dict=config.layer_dict,
    #     net_list=config.evolved_params["net"],
    #     lr_schedule=args["lr_schedule"],
    #     run_train_eval=args["run_train_eval"],
    # )
    logger.info(f"Saving parameters...")
    config.save_params_logfile()

    logger.info(f"Final test: {results}")


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
        "--folds",
        type=int,
        default=5,
        help="The number of folds for the cross validation. Default = 5.",
    )
    parser.add_argument(
        "--log_level",
        choices=["NONE", "INFO", "DEBUG"],
        default="NONE",
        help="Logging information level.",
    )

    arguments = parser.parse_args()

    main(**vars(arguments))


# from cnn.train import cross_val_train

# train_params = {
#     "batch_size": 32,
#     "epochs": 100,
#     "eval_epochs": 20,
#     "initializations": 5,
#     "folds": 5,
#     "stem_filters": 32,
#     "max_depth": 4,
#     "data_path": "spleen_dataset/data/Task09_Spleen_preprocessed/",
#     "image_size": 128,
#     "skip_slices": 0,
#     "num_channels": 1,
#     "num_classes": 2,
#     "data_augmentation": True,
# }

# unet = [
#     "vgg_n_3",
#     "vgg_d_3",
#     "vgg_d_3",
#     "vgg_d_3",
#     "vgg_d_3",
#     "vgg_u_3",
#     "vgg_u_3",
#     "vgg_u_3",
#     "vgg_u_3",
#     "vgg_n_3",
# ]

# experiment_1_8 = [
#     "vgg_d_3",
#     "vgg_d_3",
#     "vgg_n_3",
#     "ide_d",
#     "vgg_n_3",
#     "vgg_d_3",
#     "vgg_u_3",
#     "ide_d",
#     "vgg_n_3",
#     "ide_d"
# ]

# experiment_1_9 = [
#     "ide_u",
#     "vgg_d_3",
#     "vgg_d_3",
#     "ide_d",
#     "vgg_n_3",
#     "ide_d",
#     "vgg_u_3",
#     "vgg_d_3",
#     "vgg_d_3",
#     "ide_u"
# ]

# net_list = unet

# layer_dict = {
#     "vgg_d_3": {
#         "cell": "DownscalingCell",
#         "block": "VGGBlock",
#         "kernel": 3,
#         "prob": 1 / 6,
#     },
#     "vgg_u_3": {
#         "cell": "UpscalingCell",
#         "block": "VGGBlock",
#         "kernel": 3,
#         "prob": 1 / 6,
#     },
#     "vgg_n_3": {
#         "cell": "NonscalingCell",
#         "block": "VGGBlock",
#         "kernel": 3,
#         "prob": 1 / 6,
#     },
#     "ide_d": {
#         "cell": "DownscalingCell",
#         "block": "IdentityBlock",
#         "prob": 1 / 6,
#     },
#     "ide_u": {
#         "cell": "UpscalingCell",
#         "block": "IdentityBlock",
#         "prob": 1 / 6,
#     },
#     "ide_n": {
#         "cell": "NonscalingCell",
#         "block": "IdentityBlock",
#         "prob": 1 / 6,
#     },
# }

# mean_dsc, std_dsc = cross_val_train(train_params, layer_dict, net_list)

# print(f"{mean_dsc} +- {std_dsc}")
