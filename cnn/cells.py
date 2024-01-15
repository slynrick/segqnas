import importlib

from keras.initializers import HeUniform
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    concatenate,
)
from keras.regularizers import L2


class Cell(object):
    def __init__(self, block, kernel_size, filters):
        self.block = self._instatiate_block(block, kernel_size, filters)
        self.data_format = "channels_last"
        self.filters = filters
        self.initializer = HeUniform(seed=0)
        self.kernel_size = kernel_size
        self.padding = "same"
        self.regularizer = L2(1e-6)

    def _instatiate_block(self, block, kernel_size, filters):
        module = importlib.import_module("cnn.blocks")
        class_ = getattr(module, block)
        return class_(kernel_size, filters)

    def _concat(self, inputs, name=None):
        return concatenate(inputs, name=name, axis=-1)

    def _conv_1x1(self, inputs, name=None):
        return Conv2D(
            filters=self.filters,
            kernel_size=1,
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            use_bias=False,
            name=name,
        )(inputs)

    def _upsampling_conv(self, inputs, name=None):
        return Conv2DTranspose(
            filters=self.filters,
            kernel_size=3,
            strides=2,
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            use_bias=False,
            name=name,
        )(inputs)

    def _downsampling_conv(self, inputs, name=None):
        return Conv2D(
            filters=self.filters,
            kernel_size=3,
            strides=2,
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            use_bias=False,
            name=name,
        )(inputs)

    def _relu_activation(self, inputs, name=None):
        return Activation("relu", name=name)(inputs)

    def _batch_norm(self, inputs, is_train, name=None):
        return BatchNormalization(axis=-1, momentum=0.9, epsilon=2e-5, name=name)(
            inputs=inputs, training=is_train
        )

    def _concat_with_skip_connection_if_needed(self, inputs, name=None, is_train=True):
        x = inputs
        if isinstance(x, list):
            x = self._concat(x, name=f"{name}_Concatenation")
            x = self._conv_1x1(x, name=f"{name}_Convolution")
            x = self._batch_norm(x, is_train=is_train, name=f"{name}_Normalization")
            x = self._relu_activation(x, name=f"{name}_Activation")
        return x


class DownscalingCell(Cell):
    def _downsample(self, inputs, name=None, is_train=True):
        x = inputs
        x = self._downsampling_conv(x, name=f"{name}_Downsampling_Convolution")
        x = self._batch_norm(
            x, is_train=is_train, name=f"{name}_Downsampling_Normalization"
        )
        x = self._relu_activation(x, name=f"{name}_Downsampling_Activation")
        return x

    def __call__(self, inputs, name=None, is_train=True):
        x = inputs
        x = self._concat_with_skip_connection_if_needed(x, name=name, is_train=is_train)
        x = self.block(x, name=name, is_train=is_train)
        x = self._downsample(x, name=name, is_train=is_train)
        return x


class UpscalingCell(Cell):
    def _upsample(self, inputs, name=None, is_train=True):
        x = inputs
        x = self._upsampling_conv(x, name=f"{name}_Upsampling_Convolution")
        x = self._batch_norm(
            x, is_train=is_train, name=f"{name}_Upsampling_Normalization"
        )
        x = self._relu_activation(x, name=f"{name}_Upsampling_Activation")
        return x

    def __call__(self, inputs, name=None, is_train=True):
        x = inputs
        x = self._concat_with_skip_connection_if_needed(x, name=name, is_train=is_train)
        x = self.block(x, name=name, is_train=is_train)
        x = self._upsample(x, name=name, is_train=is_train)
        return x


class NonscalingCell(Cell):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs
        x = self._concat_with_skip_connection_if_needed(x, name=name, is_train=is_train)
        x = self.block(x, name=name, is_train=is_train)
        return x
