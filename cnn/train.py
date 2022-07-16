""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Train a model (single GPU).

    References:
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/cifar10_main.py

"""
import csv
import os
import platform
import time
from logging import addLevelName

import pandas as pd
import segmentation_models as sm
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop

from cnn import hparam, input, loss, model

sm.set_framework("tf.keras")


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
    tf.config.experimental.set_visible_devices(gpus[int(id_num.split("_")[-1])], "GPU")

    # filtered_dict = {key: item for key, item in fn_dict.items() if key in net_list}
    tf.compat.v1.logging.log(
        level=tf.compat.v1.logging.get_verbosity(),
        msg=f"params {params}",
    )

    hparams = hparam.HParams(**params)

    train_dataset_descriptor_filepath = os.path.join(
        hparams.descriptor_files_path,
        "train.txt",
    )

    val_dataset_descriptor_filepath = os.path.join(
        hparams.descriptor_files_path,
        "val.txt",
    )

    augmentation = None
    if hparams.data_augmentation:
        augmentation = input.get_training_augmentation(hparams.height, hparams.width)

    train_dataset = input.PascalVOC2012Dataset(
        train_dataset_descriptor_filepath,
        images_path=hparams.images_path,
        masks_path=hparams.masks_path,
        image_height=hparams.height,
        image_width=hparams.width,
        augmentation=augmentation,
        preprocessing=input.get_preprocessing(sm.get_preprocessing(hparams.backbone)),
    )

    val_dataset = input.PascalVOC2012Dataset(
        val_dataset_descriptor_filepath,
        images_path=hparams.images_path,
        masks_path=hparams.masks_path,
        image_height=hparams.height,
        image_width=hparams.width,
        preprocessing=input.get_preprocessing(sm.get_preprocessing(hparams.backbone)),
    )

    train_dataloader = input.Dataloader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True
    )
    val_dataloader = input.Dataloader(
        val_dataset, batch_size=hparams.eval_batch_size, shuffle=False
    )

    net = model.build_net(
        input_shape=(hparams.height, hparams.width, hparams.num_channels),
        num_classes=hparams.num_classes,
        fn_dict=fn_dict,
        net_list=net_list,
    )

    decay = hparams.decay if hparams.optimizer == "RMSProp" else None

    net.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=sm.losses.bce_jaccard_loss,
        metrics=[loss.MyMeanIOU(num_classes=hparams.num_classes, name="mean_iou")],
    )

    # tf.compat.v1.logging.log(
    #    level=tf.compat.v1.logging.get_verbosity(), msg=f"net {net.summary()}"
    # )
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
            epochs=hparams.max_epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", mode="min", verbose=1, patience=5
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(model_path, "best_model.h5"),
                    save_weights_only=True,
                    save_best_only=True,
                    mode="min",
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

    val_mean_iou = history.history["val_mean_iou"][-1]

    tf.compat.v1.logging.log(
        level=tf.compat.v1.logging.get_verbosity(), msg=f"val_mean_iou {val_mean_iou}"
    )

    return val_mean_iou
