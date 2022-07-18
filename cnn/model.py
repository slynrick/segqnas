""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - CNN Model.

    References:
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/cifar10.py
    https://github.com/tensorflow/models/blob/r1.10.0/official/resnet/resnet_model.py
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/model_base.py

"""
import tensorflow as tf
from tensorflow.keras import Input, Model, layers


class ConvBlock(object):
    """Convolutional Block with Conv -> BatchNorm -> ReLU"""

    def __init__(self, kernel, filters, strides, mu, epsilon):
        """Initialize ConvBlock.

        Args:
            kernel: (int) represents the size of the convolution window (3 means [3, 3]).
            filters: (int) number of filters.
            strides: (int) specifies the strides of the convolution operation (1 means [1, 1]).
            mu: (float) batch normalization mean.
            epsilon: (float) batch normalization epsilon.
        """

        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.batch_norm_mu = mu
        self.batch_norm_epsilon = epsilon
        self.activation = layers.Activation("relu")
        self.initializer = "he_normal"
        self.padding = "same"

    def __call__(self, inputs, name=None, is_train=True):
        """Convolutional block with convolution op + batch normalization op.

        Args:
            inputs: input tensor to the block.
            name: (str) name of the block.
            is_train: (bool) True if block is going to be created for training.

        Returns:
            output tensor.
        """

        tensor = self._conv2d(inputs, name=name + "_conv")
        tensor = self._batch_norm(tensor, is_train, name=name + "_bn")
        tensor = self.activation(tensor)

        return tensor

    def _conv2d(self, inputs, name=None):
        """Convolution operation wrapper.

        Args:
            inputs: input tensor to the layer.
            name: (str) name of the layer.

        Returns:
            output tensor.
        """

        return layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=None,
            padding=self.padding,
            strides=self.strides,
            data_format="channels_last",
            kernel_initializer=self.initializer,
            bias_initializer=self.initializer,
            name=name,
        )(inputs)

    def _batch_norm(self, inputs, is_train, name=None):
        """Batch normalization layer wrapper.

        Args:
            inputs: input tensor to the layer.
            is_train: (bool) True if layer is going to be created for training.
            name: (str) name of the block.

        Returns:
            output tensor.
        """

        return layers.BatchNormalization(
            axis=-1,
            momentum=self.batch_norm_mu,
            epsilon=self.batch_norm_epsilon,
            name=name,
        )(inputs=inputs, training=is_train)


class Conv2xBlock(object):
    """Convolutional Block with Conv -> BatchNorm -> ReLU"""

    def __init__(self, kernel, filters, strides, mu, epsilon):
        """Initialize ConvBlock.

        Args:
            kernel: (int) represents the size of the convolution window (3 means [3, 3]).
            filters: (int) number of filters.
            strides: (int) specifies the strides of the convolution operation (1 means [1, 1]).
            mu: (float) batch normalization mean.
            epsilon: (float) batch normalization epsilon.
        """

        self.kernel_size = kernel
        self.filters = filters
        self.strides = strides
        self.batch_norm_mu = mu
        self.batch_norm_epsilon = epsilon
        self.activation = layers.Activation("relu")
        self.initializer = "he_normal"
        self.padding = "same"

    def __call__(self, inputs, name=None, is_train=True):
        """Convolutional block with convolution op + batch normalization op.

        Args:
            inputs: input tensor to the block.
            name: (str) name of the block.
            is_train: (bool) True if block is going to be created for training.

        Returns:
            output tensor.
        """

        tensor = self._conv2d(inputs, name=name + "_conv")
        tensor = self._batch_norm(tensor, is_train, name=name + "_bn")
        tensor = layers.Activation("relu")(tensor)
        tensor = self._conv2d(tensor, name=name + "_conv_2")
        tensor = self._batch_norm(tensor, is_train, name=name + "_bn_2")
        tensor = layers.Activation("relu")(tensor)

        return tensor

    def _conv2d(self, inputs, name=None):
        """Convolution operation wrapper.

        Args:
            inputs: input tensor to the layer.
            name: (str) name of the layer.

        Returns:
            output tensor.
        """

        return layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=None,
            padding=self.padding,
            strides=self.strides,
            data_format="channels_last",
            kernel_initializer=self.initializer,
            bias_initializer=self.initializer,
            name=name,
        )(inputs)

    def _batch_norm(self, inputs, is_train, name=None):
        """Batch normalization layer wrapper.

        Args:
            inputs: input tensor to the layer.
            is_train: (bool) True if layer is going to be created for training.
            name: (str) name of the block.

        Returns:
            output tensor.
        """

        return layers.BatchNormalization(
            axis=-1,
            momentum=self.batch_norm_mu,
            epsilon=self.batch_norm_epsilon,
            name=name,
        )(inputs=inputs, training=is_train)


class NoOp(object):
    pass


encoder_feature_layers = {
    "efficientnetb0": (
        "block6a_expand_activation",
        "block4a_expand_activation",
        "block3a_expand_activation",
        "block2a_expand_activation",
    ),
    "efficientnetb1": (
        "block6a_expand_activation",
        "block4a_expand_activation",
        "block3a_expand_activation",
        "block2a_expand_activation",
    ),
    "efficientnetb2": (
        "block6a_expand_activation",
        "block4a_expand_activation",
        "block3a_expand_activation",
        "block2a_expand_activation",
    ),
    "efficientnetb3": (
        "block6a_expand_activation",
        "block4a_expand_activation",
        "block3a_expand_activation",
        "block2a_expand_activation",
    ),
    "efficientnetb4": (
        "block6a_expand_activation",
        "block4a_expand_activation",
        "block3a_expand_activation",
        "block2a_expand_activation",
    ),
    "efficientnetb5": (
        "block6a_expand_activation",
        "block4a_expand_activation",
        "block3a_expand_activation",
        "block2a_expand_activation",
    ),
    "efficientnetb6": (
        "block6a_expand_activation",
        "block4a_expand_activation",
        "block3a_expand_activation",
        "block2a_expand_activation",
    ),
    "efficientnetb7": (
        "block6a_expand_activation",
        "block4a_expand_activation",
        "block3a_expand_activation",
        "block2a_expand_activation",
    ),
}


def build_net(
    input_shape, num_classes, fn_dict, net_list, is_train=True, mu=0.9, epsilon=2e-5
):

    layer_dict = {}
    for name, definition in fn_dict.items():
        if definition["function"] in ["Conv2xBlock"]:
            definition["params"]["mu"] = mu
            definition["params"]["epsilon"] = epsilon
        layer_dict[name] = globals()[definition["function"]](**definition["params"])

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=1000,
    )

    skips = [
        backbone.get_layer(name=layer_name).output
        for layer_name in encoder_feature_layers["efficientnetb0"]
    ]

    input = backbone.input
    x = backbone.output

    for i, f in enumerate(net_list):
        x = layers.UpSampling2D(size=2)(x)

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        if skip is not None:
            x = layers.Concatenate(axis=3)([x, skip])

        x = layer_dict[f](inputs=x, name=f"{f}_{i}", is_train=is_train)

    x = layers.Conv2D(
        filters=num_classes,
        kernel_size=(3, 3),
        padding="same",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        name="final_conv",
    )(x)

    x = layers.Activation("softmax", name="softmax")(x)

    # create keras model instance
    model = Model(input, x)
    return model
