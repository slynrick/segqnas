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

    gpu_id = int(id_num.split("_")[-1])%len(gpus)

    tf.config.experimental.set_visible_devices(gpus[gpu_id], "GPU")
    
    # try:
    #     for gpu in gpus:
    #         tf.config.experimental.set_memory_growth(gpu, True)
    # except RuntimeError as e:
    #     print(e)

    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[gpu_id], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20480)])
    except RuntimeError as e:
        print(e)

    # filtered_dict = {key: item for key, item in fn_dict.items() if key in net_list}
    tf.compat.v1.logging.log(
        level=tf.compat.v1.logging.get_verbosity(),
        msg=f"params {params}",
    )

    hparams = hparam.HParams(**params)

    seed_value= 0

    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    data_path = hparams.data_path
    num_classes = hparams.num_classes
    num_channels = hparams.num_channels
    image_size = hparams.image_size
    batch_size = hparams.batch_size
    epochs = hparams.epochs

    train_images_filepaths, val_images_filepaths, train_labels_filepaths, val_labels_filepaths = input.get_train_val_filenames(data_path)

    train_dataset = input.SpleenDataset(train_images_filepaths, train_labels_filepaths)
    val_dataset = input.SpleenDataset(val_images_filepaths, val_labels_filepaths)

    train_dataloader = input.Dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = input.Dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    net = model.build_net((image_size, image_size, num_channels), num_classes, fn_dict=fn_dict, net_list=net_list)

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
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(model_path, "best_model.h5"),
                    save_weights_only=True,
                    save_best_only=True,
                    mode="min",
                    monitor="val_loss",
                ),
                tf.keras.callbacks.ReduceLROnPlateau(),
            ],
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
        level=tf.compat.v1.logging.get_verbosity(), msg=f"val_gen_dice_coef {val_gen_dice_coef}"
    )

    return val_gen_dice_coef
