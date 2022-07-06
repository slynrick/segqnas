""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - CNN Model.

    References:
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/cifar10.py
    https://github.com/tensorflow/models/blob/r1.10.0/official/resnet/resnet_model.py
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/model_base.py

"""
import tensorflow as tf
from tensorflow.keras import Input, Model, initializers, layers


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
        self.activation = layers.Activation('relu')
        self.initializer = initializers.HeNormal
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

        tensor = self._conv2d(inputs, name=name + '_conv')
        tensor = self._batch_norm(tensor, is_train, name=name + '_bn')
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

        return layers.Conv2D(filters=self.filters,
                            kernel_size=self.kernel_size,
                            activation=None,
                            padding=self.padding,
                            strides=self.strides,
                            data_format='channels_last',
                            kernel_initializer=self.initializer(),
                            bias_initializer=self.initializer(), name=name)(inputs)


    def _batch_norm(self, inputs, is_train, name=None):
        """Batch normalization layer wrapper.

        Args:
            inputs: input tensor to the layer.
            is_train: (bool) True if layer is going to be created for training.
            name: (str) name of the block.

        Returns:
            output tensor.
        """

        return layers.BatchNormalization(axis=-1,
                                        momentum=self.batch_norm_mu,
                                        epsilon=self.batch_norm_epsilon,
                                        name=name)(inputs=inputs, training=is_train)

class MaxPooling(object):
    def __init__(self, kernel, strides):
        """Initialize MaxPooling.

        Args:
            kernel: (int) represents the size of the pooling window (3 means [3, 3]).
            strides: (int) specifies the strides of the pooling operation (1 means [1, 1]).
        """

        self.pool_size = kernel
        self.strides = strides
        self.padding = "VALID"

    def __call__(self, inputs, name=None):
        """Create Max Pooling layer.

        Args:
            inputs: input tensor to the layer.
            name: (str) name of the layer.

        Returns:
            output tensor.
        """

        # check of the image size
        if inputs.shape[2] > 1:
            return layers.MaxPooling2D(pool_size=self.pool_size,
                                        strides=self.strides,
                                        data_format='channels_last',
                                        padding=self.padding,
                                        name=name)(inputs)
            
        else:
            return inputs


class AvgPooling(object):
    def __init__(self, kernel, strides):
        """Initialize AvgPooling.

        Args:
            kernel: (int) represents the size of the pooling window (3 means [3, 3]).
            strides: (int) specifies the strides of the pooling operation (1 means [1, 1]).
        """

        self.pool_size = kernel
        self.strides = strides
        self.padding = "VALID"

    def __call__(self, inputs, name=None):
        """Create Average Pooling layer.

        Args:
            inputs: input tensor to the layer.
            name: (str) name of the layer.

        Returns:
            output tensor.
        """

        # check of the image size
        if inputs.shape[2] > 1:
            return layers.AveragePooling2D(pool_size=self.pool_size,
                                            strides=self.strides,
                                            data_format='channels_last',
                                            padding=self.padding,
                                            name=name)(inputs)
        else:
            return inputs


class NoOp(object):
    pass


def get_segmentation_model(input_shape, num_classes, fn_dict, net_list, is_train=True, mu=0.9, epsilon=2e-5):

    layer_dict = {}
    for name, definition in fn_dict.items():
        if definition["function"] in ["ConvBlock"]:
            definition["params"]["mu"] = mu
            definition["params"]["epsilon"] = epsilon
        layer_dict[name] = globals()[definition["function"]](
            **definition["params"]
        )
    
    skip_connections = []
    inputs = Input(shape=input_shape)
    x = inputs

    for i, f in enumerate(net_list):
        if f == "no_op":
                continue
        elif isinstance(layer_dict[f], ConvBlock):
            x = layer_dict[f](
                inputs=x, name=f"l{i}_{f}", is_train=is_train
            )
        else:
            skip_connections.append(x)
            x = layer_dict[f](inputs=x, name=f"l{i}_{f}")
    
    for i, f in enumerate(net_list[::-1]):
        if f == "no_op":
            continue
        elif isinstance(layer_dict[f], ConvBlock):
            x = layer_dict[f](
                inputs=x, name=f"l{i+len(net_list)}_{f}", is_train=is_train
            )
        else:
            x = layers.UpSampling2D(
                size=(2, 2), data_format="channels_last", name=f"l{i+len(net_list)}_upsampling"
            )(x)
            x = layers.Concatenate()([x, skip_connections.pop()])

    outputs = layers.Conv2D(
        filters=num_classes,
        kernel_size=1,
        activation=None,
        padding="same",
        strides=1,
        data_format="channels_last",
        kernel_initializer="he_normal",
        bias_initializer="he_normal",
        name="final_conv",
    )(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
