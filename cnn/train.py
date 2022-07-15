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

from cnn import hparam, input, model
from cnn.hooks import GetBestHook, TimeOutHook

sm.set_framework("tf.keras")

# TRAIN_TIMEOUT = 5400
TRAIN_TIMEOUT = 86400


class MyMeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(
            tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight
        )


def _model_fn(features, labels, mode, params):
    """Returns a function that will build the model.

    Args:
        features: a tensor with a batch of features.
        labels: a tensor with a batch of labels (masks).
        mode: ModeKeys.TRAIN or EVAL.
        params: tf.contrib.training.HParams object with various hyperparameters.

    Returns:
          A EstimatorSpec object.
    """

    is_train = mode == tf.estimator.ModeKeys.TRAIN

    with tf.compat.v1.variable_scope("q_net"):
        loss, grads_and_vars, predictions = _get_loss_and_grads(
            is_train=is_train, params=params, features=features, labels=labels
        )
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

    tf.summary.scalar("train_loss", loss)

    decay = params.decay if params.optimizer == "RMSProp" else None
    optimizer = _optimizer(
        params.optimizer, params.learning_rate, params.momentum, decay
    )

    train_hooks = _train_hooks(params)

    # Create single grouped train op
    train_op = [
        optimizer.apply_gradients(
            grads_and_vars, global_step=tf.compat.v1.train.get_global_step()
        )
    ]
    train_op.extend(update_ops)
    train_op = tf.group(*train_op)

    metrics = {
        "mean_iou": tf.compat.v1.metrics.mean_iou(
            tf.expand_dims(tf.argmax(input=labels, axis=-1), -1),
            predictions["classes"],
            predictions["masks"].shape[-1],
        )
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=train_hooks,
        eval_metric_ops=metrics,
    )


def _optimizer(optimizer_name, learning_rate, momentum, decay):
    """Create optimizer defined by *optimizer_name*.

    Args:
        optimizer_name: (str) one of 'RMSProp' or 'Momentum'.
        learning_rate: (float) learning rate for the optimizer.
        momentum: (float) momentum for the optimizer.
        decay: (float) RMSProp decay; only necessary when using the RMSProp optimizer.

    Returns:
        optimizer and list of training hooks.
    """

    if optimizer_name == "RMSProp":
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate, decay=decay, momentum=momentum
        )
    else:
        # TODO
        optimizer = tf.compat.v1.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=momentum
        )
    return optimizer


def _train_hooks(params):
    """Create training hooks for timeout and logging. The variables to be logged during
        training depend on the optimizer defined by *params.optimizer*.

    Args:
        params: tf.contrib.training.HParams object with various hyperparameters.

    Returns:
        list of training hooks.
    """

    lr = tf.constant(params.learning_rate)
    momentum = tf.constant(params.momentum)
    w_decay = tf.constant(params.weight_decay)

    if params.optimizer == "RMSProp":
        decay = tf.constant(params.decay)
        tensors_to_log = {
            "decay": decay,
            "learning_rate": lr,
            "momentum": momentum,
            "weight_decay": w_decay,
        }
    else:
        tensors_to_log = {
            "learning_rate": lr,
            "momentum": momentum,
            "weight_decay": w_decay,
        }

    timeout_hook = TimeOutHook(
        timeout_sec=TRAIN_TIMEOUT, t0=params.t0, every_n_steps=100
    )
    logging_hook = tf.compat.v1.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100
    )

    train_hooks = [logging_hook, timeout_hook]

    return train_hooks


def _get_loss_and_grads(is_train, params, features, labels):
    """Create model defined by *params.net_list*, get its loss and gradients.

    Args:
        is_train: (bool) True if the graph os for training.
        features: a Tensor with features.
        labels: a Tensor with labels corresponding to *features*.
        params: tf.contrib.training.HParams object with various hyperparameters.

    Returns:
        A tuple containing: the loss, the list for gradients with respect to each variable in
        the model, and predictions.
    """

    logits = params.net.create_network(
        inputs=features, net_list=params.net_list, is_train=is_train
    )

    # predictions = {'masks': tf.argmax(input=pred_masks, axis=1),
    #               'probabilities': tf.nn.softmax(pred_masks, name='softmax_tensor')}
    predictions = {
        "classes": tf.expand_dims(tf.argmax(input=logits, axis=-1), -1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    one_hot_mask = []
    for _class in range(21):
        class_mask = tf.reduce_all(tf.equal(predictions["classes"], _class), axis=-1)
        one_hot_mask.append(class_mask)
    one_hot_mask = tf.stack(one_hot_mask, axis=-1)
    one_hot_mask = tf.cast(one_hot_mask, tf.float32)

    predictions["masks"] = one_hot_mask

    # loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
    loss = tf.keras.losses.BinaryCrossentropy()(y_true=labels, y_pred=logits)
    # loss = loss_function.DiceLoss()(y_true=labels, y_pred=logits)

    # Apply weight decay for every trainable variable in the model
    model_params = tf.compat.v1.trainable_variables()
    loss += params.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in model_params])

    gradients = tf.gradients(loss, model_params)

    return loss, list(zip(gradients, model_params)), predictions


def train_and_eval(params, run_config, train_input_fn, eval_input_fn):
    """Train a model and evaluate it for the last *params.epochs_to_eval*. Return the maximum
        mean iou.

    Args:
        params: tf.contrib.training.HParams object with various hyperparameters.
        run_config: tf.Estimator.RunConfig object.
        train_input_fn: input_fn for training.
        eval_input_fn: input_fn for evaluation.

    Returns:
        maximum mean iou.
    """

    # best_mean_iou[0] --> best mean iou in the last epochs; best_mean_iou[1] --> corresponding step
    best_mean_iou = [0, 0]

    # Calculate max_steps based on epochs_to_eval.
    train_steps = params.max_steps - params.epochs_to_eval * int(params.steps_per_epoch)

    # Create estimator.
    segmentation_model = tf.estimator.Estimator(
        model_fn=_model_fn, config=run_config, params=params
    )

    # Train estimator for the first train_steps.
    segmentation_model.train(input_fn=train_input_fn, max_steps=train_steps)

    eval_hook = GetBestHook(name="mean_iou/Select_1:0", best_metric=best_mean_iou)

    # Run the last steps_to_eval to complete training and also record validation mean iou.
    # Evaluate 1 time per epoch.
    for _ in range(params.epochs_to_eval):
        train_steps += int(params.steps_per_epoch)
        segmentation_model.train(input_fn=train_input_fn, max_steps=train_steps)

        segmentation_model.evaluate(
            input_fn=eval_input_fn, steps=None, hooks=[eval_hook]
        )

    return best_mean_iou[0]


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

    # net = sm.Unet(
    #     hparams.backbone,
    #     classes=hparams.num_classes,
    #     input_shape=(hparams.height, hparams.width, hparams.num_channels),
    #     activation="softmax",
    #     encoder_weights="imagenet",
    # )

    net = model.build_net(
        input_shape=(hparams.height, hparams.width, hparams.num_channels),
        num_classes=hparams.num_classes,
        fn_dict=fn_dict,
        net_list=net_list,
    )

    decay = hparams.decay if hparams.optimizer == "RMSProp" else None
    optimizer = _optimizer(
        hparams.optimizer, hparams.learning_rate, hparams.momentum, decay
    )

    net.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=sm.losses.bce_jaccard_loss,
        metrics=[MyMeanIOU(num_classes=hparams.num_classes, name="mean_iou")],
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
    with open(net_list_file_path, mode="wb") as f:
        write = csv.writer(f)
        write.writerows(net_list)

    val_mean_iou = history.history["val_mean_iou"][-1]

    tf.compat.v1.logging.log(
        level=tf.compat.v1.logging.get_verbosity(), msg=f"val_mean_iou {val_mean_iou}"
    )

    return val_mean_iou
