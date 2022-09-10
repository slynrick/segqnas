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
from spleen_dataset.config import dataset_folder
from spleen_dataset.dataloader import (SpleenDataloader, SpleenDataset,
                                       get_training_augmentation,
                                       get_validation_augmentation)
from spleen_dataset.utils import get_list_of_patients, get_split_deterministic
from tensorflow.keras.optimizers import RMSprop

from cnn import hparam, input, loss, model


def fitness_calculation(id_num, params, fn_dict, net_list):
    """Train and evaluate a model using evolved parameters.

    Args:
        id_num: string identifying the generation number and the individual number.
        params: dictionary with parameters necessary for training, including the evolved
            hyperparameters.
        fn_dict: dict with definitions of the possible layers (name and parameters).
        net_list: list with names of layers defining the network, in the order they appear.

    Returns:
        mean iou of the model for the validation set.
    """

    os.environ["TF_SYNC_ON_FINISH"] = "0"
    os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"
    if params["log_level"] == "INFO":
        addLevelName(25, "INFO1")
        tf.compat.v1.logging.set_verbosity(25)
    elif params["log_level"] == "DEBUG":
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    model_path = os.path.join(params["experiment_path"], id_num)
    os.mkdir(model_path)

    gpus = tf.config.experimental.list_physical_devices("GPU")

    gpu_id = int(id_num.split("_")[-1]) % len(gpus)

    tf.config.experimental.set_visible_devices(gpus[gpu_id], "GPU")

    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[gpu_id],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)],
        )
    except RuntimeError as e:
        print(e)

    hparams = hparam.HParams(**params)

    data_path = hparams.data_path
    num_classes = hparams.num_classes
    num_channels = hparams.num_channels
    image_size = hparams.image_size
    batch_size = hparams.batch_size
    epochs = hparams.epochs

    patients = get_list_of_patients(dataset_folder)
    patch_size = (image_size, image_size)

    train_augmentation = get_training_augmentation(patch_size)
    val_augmentation = get_validation_augmentation(patch_size)

    num_splits = 5
    num_initializations = 3
    metric_epochs = 10

    # Training time start counting here. It needs to be defined outside model_fn(), to make it
    # valid in the multiple calls to segmentation_model.train(). Otherwise, it would be restarted.
    params["t0"] = time.time()

    node = platform.uname()[1]

    tf.compat.v1.logging.log(
        level=tf.compat.v1.logging.get_verbosity(),
        msg=f"I am node {node}! Running fitness calculation of {id_num} with "
        f"structure:\n{net_list}",
    )

    val_gen_dice_coef_list = []

    for initialization in range(num_initializations):

        for fold in range(num_splits):

            net = model.build_net(
                (image_size, image_size, num_channels),
                num_classes,
                fn_dict=fn_dict,
                net_list=net_list,
            )

            train_patients, val_patients = get_split_deterministic(
                patients, fold=fold, num_splits=num_splits, random_state=initialization
            )

            train_dataset = SpleenDataset(train_patients, only_non_empty_slices=True)
            val_dataset = SpleenDataset(val_patients, only_non_empty_slices=True)

            train_dataloader = SpleenDataloader(
                train_dataset, batch_size, train_augmentation
            )
            val_dataloader = SpleenDataloader(val_dataset, batch_size, val_augmentation)

            checkpoint_filepath = "/tmp/checkpoint"
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor="val_gen_dice_coef",
                mode="max",
                save_best_only=True,
            )

            history = net.fit(
                train_dataloader,
                validation_data=val_dataloader,
                epochs=epochs,
                verbose=1,
                callbacks=[model_checkpoint_callback],
            )

            net.load_weights(checkpoint_filepath)

            for patient in val_patients:
                patient_dataset = SpleenDataset([patient], only_non_empty_slices=True)
                patient_dataloader = SpleenDataloader(
                    patient_dataset, 1, val_augmentation, shuffle=False
                )
                results = model.evaluate(patient_dataloader)
                val_gen_dice_coef_patient = results[-1]
                val_gen_dice_coef_list.append(val_gen_dice_coef_patient)

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

    params["net"] = net
    params["net_list"] = net_list

    return mean_val_gen_dice_coef


def fitness_calculation_old(id_num, params, fn_dict, net_list):
    """Train and evaluate a model using evolved parameters.

    Args:
        id_num: string identifying the generation number and the individual number.
        params: dictionary with parameters necessary for training, including the evolved
            hyperparameters.
        fn_dict: dict with definitions of the possible layers (name and parameters).
        net_list: list with names of layers defining the network, in the order they appear.

    Returns:
        mean iou of the model for the validation set.
    """

    os.environ["TF_SYNC_ON_FINISH"] = "0"
    os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"
    if params["log_level"] == "INFO":
        addLevelName(25, "INFO1")
        tf.compat.v1.logging.set_verbosity(25)
    elif params["log_level"] == "DEBUG":
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    model_path = os.path.join(params["experiment_path"], id_num)
    os.mkdir(model_path)

    gpus = tf.config.experimental.list_physical_devices("GPU")

    gpu_id = int(id_num.split("_")[-1]) % len(gpus)

    tf.config.experimental.set_visible_devices(gpus[gpu_id], "GPU")

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

    # try:
    #     tf.config.experimental.set_virtual_device_configuration(gpus[gpu_id], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20480)])
    # except RuntimeError as e:
    #     print(e)

    hparams = hparam.HParams(**params)

    seed_value = 0

    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    data_path = hparams.data_path
    num_classes = hparams.num_classes
    num_channels = hparams.num_channels
    image_size = hparams.image_size
    batch_size = hparams.batch_size
    epochs = hparams.epochs

    (
        train_images_filepaths,
        val_images_filepaths,
        train_labels_filepaths,
        val_labels_filepaths,
    ) = input.get_train_val_filenames(data_path)

    train_dataset = input.SpleenDataset(train_images_filepaths, train_labels_filepaths)
    val_dataset = input.SpleenDataset(val_images_filepaths, val_labels_filepaths)

    train_dataloader = input.Dataloader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = input.Dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    net = model.build_net(
        (image_size, image_size, num_channels),
        num_classes,
        fn_dict=fn_dict,
        net_list=net_list,
    )

    params["net"] = net
    params["net_list"] = net_list

    # Training time start counting here. It needs to be defined outside model_fn(), to make it
    # valid in the multiple calls to segmentation_model.train(). Otherwise, it would be restarted.
    params["t0"] = time.time()

    node = platform.uname()[1]

    tf.compat.v1.logging.log(
        level=tf.compat.v1.logging.get_verbosity(),
        msg=f"I am node {node}! Running fitness calculation of {id_num} with "
        f"structure:\n{net_list}",
    )

    try:
        history = net.fit(
            train_dataloader,
            validation_data=val_dataloader,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", mode="min", verbose=1, patience=5
                ),
                tf.keras.callbacks.ReduceLROnPlateau(),
            ],
            verbose=2,
        )
    except tf.errors.ResourceExhaustedError:
        tf.compat.v1.logging.log(
            level=tf.compat.v1.logging.get_verbosity(),
            msg=f"Model is probably too large... Resource Exhausted Error!",
        )
        return 0

    # save csv file with learning curve (history)
    history_df = pd.DataFrame(history.history)
    history_csv_file_path = os.path.join(model_path, "history.csv")
    with open(history_csv_file_path, mode="wb") as f:
        history_df.to_csv(f)

    # save net list as csv (layers)
    net_list_file_path = os.path.join(model_path, "net_list.csv")
    with open(net_list_file_path, mode="w") as f:
        write = csv.writer(f)
        write.writerow(net_list)

    val_gen_dice_coef = history.history["val_gen_dice_coef"][-1]

    tf.compat.v1.logging.log(
        level=tf.compat.v1.logging.get_verbosity(),
        msg=f"val_gen_dice_coef {val_gen_dice_coef}",
    )

    return val_gen_dice_coef
