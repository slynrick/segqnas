""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Train a model (single GPU).

    References:
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/cifar10_main.py

"""
import csv
import os
import platform
import random
import time
from logging import addLevelName

import numpy as np
import pandas as pd
import tensorflow as tf
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from spleen_dataset.config import dataset_folder
from spleen_dataset.dataloader import (
    SpleenDataloader,
    SpleenDataset,
    get_training_augmentation,
    get_validation_augmentation,
)
from spleen_dataset.utils import get_list_of_patients, get_split_deterministic
from tensorflow.keras.optimizers import RMSprop

from cnn import input, loss, model


def fitness_calculation(id_num, train_params, layer_dict, net_list, cell_list=None):
    """Train and evaluate a model using evolved parameters.

    Args:
        id_num: string identifying the generation number and the individual number.
        train_params: dictionary with parameters necessary for training
        layer_dict: dict with definitions of the possible layers (name and parameters).
        net_list: list with names of layers defining the network, in the order they appear.
        cell_list: list of predefined cell types that defined a topology (if provided).

    Returns:
        Mean dice coeficient of the model for the last 20% epochs for 3 times 5-fold cross validation.
    """

    print(train_params)

    os.environ["TF_SYNC_ON_FINISH"] = "0"
    os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"
    if train_params["log_level"] == "INFO":
        addLevelName(25, "INFO1")
        tf.compat.v1.logging.set_verbosity(25)
    elif train_params["log_level"] == "DEBUG":
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    model_path = os.path.join(train_params["experiment_path"], id_num)
    maybe_mkdir_p(model_path)

    gpus = tf.config.experimental.list_physical_devices("GPU")

    gpu_id = int(id_num.split("_")[-1]) % len(gpus)

    tf.config.experimental.set_visible_devices(gpus[gpu_id], "GPU")

    if len(gpus) > 1:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[gpu_id],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)],
            )
        except RuntimeError as e:
            print(e)

    data_path = train_params["data_path"]
    num_classes = train_params["num_classes"]
    num_channels = train_params["num_channels"]
    image_size = train_params["image_size"]
    batch_size = train_params["batch_size"]
    epochs = train_params["epochs"]
    num_folds = train_params["folds"]
    num_initializations = train_params["initializations"]

    patients = get_list_of_patients(dataset_folder)
    patch_size = (image_size, image_size)

    train_augmentation = get_training_augmentation(patch_size)
    val_augmentation = get_validation_augmentation(patch_size)

    # Training time start counting here. It needs to be defined outside model_layer(), to make it
    # valid in the multiple calls to segmentation_model.train(). Otherwise, it would be restarted.
    train_params["t0"] = time.time()

    node = platform.uname()[1]

    tf.compat.v1.logging.log(
        level=tf.compat.v1.logging.get_verbosity(),
        msg=f"I am node {node}! Running fitness calculation of {id_num} with "
        f"structure:\n{net_list}",
    )

    val_gen_dice_coef_list = []
    evaluation_epochs = int(0.2 * epochs)

    try:
        for initialization in range(num_initializations):
            for fold in range(num_folds):
                net = model.build_net(
                    (image_size, image_size, num_channels),
                    num_classes,
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

                train_dataset = SpleenDataset(
                    train_patients, only_non_empty_slices=True
                )

                val_dataset = SpleenDataset(val_patients, only_non_empty_slices=True)
                train_dataloader = SpleenDataloader(
                    train_dataset, batch_size, train_augmentation
                )

                val_dataloader = SpleenDataloader(
                    val_dataset, batch_size, val_augmentation
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

                val_gen_dice_coef_list.extend(
                    history.history["val_gen_dice_coef"][-evaluation_epochs:]
                )

                tf.compat.v1.logging.log(
                    level=tf.compat.v1.logging.get_verbosity(),
                    msg=f"DSC of last {evaluation_epochs} epochs of {id_num}: {history.history['val_gen_dice_coef'][-evaluation_epochs:]}",
                )

    except Exception as e:
        tf.compat.v1.logging.log(
            level=tf.compat.v1.logging.get_verbosity(),
            msg=f"Exception: {e}",
        )

        return 0

    mean_val_gen_dice_coef = np.mean(val_gen_dice_coef_list)
    std_val_gen_dice_coef = np.std(val_gen_dice_coef_list)
    tf.compat.v1.logging.log(
        level=tf.compat.v1.logging.get_verbosity(),
        msg=f"val_gen_dice_coef {mean_val_gen_dice_coef} +- {std_val_gen_dice_coef}",
    )

    # save net list as csv (layers)
    net_list_file_path = os.path.join(model_path, "net_list.csv")

    with open(net_list_file_path, mode="w") as f:
        write = csv.writer(f)
        write.writerow(net_list)

    train_params["net"] = net
    train_params["net_list"] = net_list

    return mean_val_gen_dice_coef
