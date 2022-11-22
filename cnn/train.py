""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Train a model (single GPU).
"""
import csv
import os
import platform
import time
from logging import addLevelName

import numpy as np
import tensorflow as tf
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

# from spleen_dataset.config import dataset_folder as spleen_dataset_folder
# from spleen_dataset.dataloader import (
#     SpleenDataloader,
#     SpleenDataset,
#     get_training_augmentation as get_spleen_training_augmentation,
#     get_validation_augmentation as get_spleen_validation_augmentation,
# )
# from spleen_dataset.utils import (
#     get_list_of_patients as get_list_of_spleen_patients,
#     get_split_deterministic as get_spleen_split_deterministic,
# )

from cnn.input import (
    get_list_of_patients,
    get_training_augmentation,
    get_validation_augmentation,
    Dataset,
    Dataloader,
    get_split_deterministic,
)

# from prostate_dataset.config import dataset_folder as prostate_dataset_folder
# from prostate_dataset.dataloader import (
#     ProstateDataloader,
#     ProstateDataset,
#     get_training_augmentation as get_prostate_training_augmentation,
#     get_validation_augmentation as get_prostate_validation_augmentation,
# )
# from prostate_dataset.utils import (
#     get_list_of_patients as get_list_of_prostate_patients,
#     get_split_deterministic as get_prostate_split_deterministic,
# )

from cnn import model


def cross_val_train(train_params, layer_dict, net_list, cell_list=None):

    data_path = train_params["data_path"]
    num_classes = train_params["num_classes"]
    num_channels = train_params["num_channels"]
    skip_slices = train_params["skip_slices"]
    image_size = train_params["image_size"]
    batch_size = train_params["batch_size"]
    epochs = train_params["epochs"]
    eval_epochs = train_params["eval_epochs"]
    num_folds = train_params["folds"]
    num_initializations = train_params["initializations"]
    stem_filters = train_params["stem_filters"]
    max_depth = train_params["max_depth"]

    patch_size = (image_size, image_size, num_channels)

    patients = get_list_of_patients(data_path)
    train_augmentation = get_training_augmentation(patch_size)
    val_augmentation = get_validation_augmentation(patch_size)

    val_gen_dice_coef_list = []

    for initialization in range(num_initializations):
        for fold in range(num_folds):

            net = model.build_net(
                input_shape=patch_size,
                num_classes=num_classes,
                stem_filters=stem_filters,
                max_depth=max_depth,
                layer_dict=layer_dict,
                net_list=net_list,
                cell_list=cell_list,
            )

            train_patients, val_patients = get_split_deterministic(
                patients,
                fold=fold,
                num_splits=num_folds,
                random_state=initialization,
            )

            train_dataset = Dataset(
                data_path=data_path,
                patients=train_patients,
                only_non_empty_slices=True,
            )

            val_dataset = Dataset(
                data_path=data_path,
                patients=val_patients,
                only_non_empty_slices=True,
            )

            train_dataloader = Dataloader(
                dataset=train_dataset,
                batch_size=batch_size,
                skip_slices=skip_slices,
                augmentation=train_augmentation,
                shuffle=True,
            )

            val_dataloader = Dataloader(
                dataset=val_dataset,
                batch_size=batch_size,
                skip_slices=0,
                augmentation=val_augmentation,
                shuffle=False,
            )

            def learning_rate_fn(epoch):
                initial_learning_rate = 1e-3
                end_learning_rate = 1e-4
                power = 0.9
                return (
                    (initial_learning_rate - end_learning_rate)
                    * (1 - epoch / float(epochs)) ** (power)
                ) + end_learning_rate

            lr_callback = tf.keras.callbacks.LearningRateScheduler(
                learning_rate_fn, verbose=False
            )

            history = net.fit(
                train_dataloader,
                validation_data=val_dataloader,
                epochs=epochs,
                verbose=0,
                callbacks=[lr_callback],
            )

            history_eval_epochs = history.history["val_gen_dice_coef"][-eval_epochs:]

            val_gen_dice_coef_list.extend(history_eval_epochs)

            mean_dsc = np.mean(val_gen_dice_coef_list)
            std_dsc = np.std(val_gen_dice_coef_list)

    mean_dsc = np.mean(val_gen_dice_coef_list)
    std_dsc = np.std(val_gen_dice_coef_list)

    return mean_dsc, std_dsc


def fitness_calculation(id_num, train_params, layer_dict, net_list, cell_list=None):
    """Train and evaluate a model using evolved parameters.

    Args:
        id_num: string identifying the generation number and the individual number.
        train_params: dictionary with parameters necessary for training
        layer_dict: dict with definitions of the possible layers (name and parameters).
        net_list: list with names of layers defining the network, in the order they appear.
        cell_list: list of predefined cell types that defined a topology (if provided).

    Returns:
        Mean dice coeficient of the model for the last epochs for <initializations> times <folds>-fold cross validation.
    """

    os.environ["TF_SYNC_ON_FINISH"] = "0"
    os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"
    if train_params["log_level"] == "INFO":
        addLevelName(25, "INFO1")
        tf.compat.v1.logging.set_verbosity(25)
    elif train_params["log_level"] == "DEBUG":
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    model_path = os.path.join(train_params["experiment_path"], id_num)
    maybe_mkdir_p(model_path)

    # save net list as csv (layers)
    net_list_file_path = os.path.join(model_path, "net_list.csv")

    gpus = tf.config.experimental.list_physical_devices("GPU")

    if gpus:
        gpu_id = int(id_num.split("_")[-1]) % len(gpus)
        tf.config.experimental.set_visible_devices(gpus[gpu_id], "GPU")
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[gpu_id],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)],
            )
        except RuntimeError as e:
            print(e)

    train_params["t0"] = time.time()

    node = platform.uname()[1]

    tf.compat.v1.logging.log(
        level=tf.compat.v1.logging.get_verbosity(),
        msg=f"I am node {node}! Running fitness calculation of {id_num} with "
        f"structure:\n{net_list}",
    )

    try:
        mean_dsc, std_dsc = cross_val_train(train_params, layer_dict, net_list, cell_list)
    except Exception as e:
        tf.compat.v1.logging.log(
            level=tf.compat.v1.logging.get_verbosity(),
            msg=f"Exception: {e}",
        )

        return 0

    tf.compat.v1.logging.log(
        level=tf.compat.v1.logging.get_verbosity(),
        msg=f"[{id_num}] DSC: {mean_dsc} +- {std_dsc}",
    )

    with open(net_list_file_path, mode="w") as f:
        write = csv.writer(f)
        write.writerow(net_list)

    #train_params["net"] = net
    train_params["net_list"] = net_list

    return mean_dsc
