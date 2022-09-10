import torch
import torch.nn as nn
from tensorflow import keras as K


class Block(object):
    def __init__(self, kernel_size, filters, name=None):
        self.kernel_size = kernel_size
        self.filters = filters
        self.initializer = "he_normal"
        self.padding = "same"
        self.data_format = "channels_last"

    def _add(self, inputs, name=None):
        return K.layers.Add(name=name)(inputs)

    def _batch_norm(self, inputs, is_train, name=None):
        return K.layers.BatchNormalization(
            axis=-1, momentum=0.9, epsilon=2e-5, name=name
        )(inputs=inputs, training=is_train)

    def _relu_activation(self, inputs):
        return K.layers.Activation("relu")(inputs)

    def _relu6_activation(self, inputs):
        return K.layers.ReLU(max_value=6.0)(inputs)

    def _linear_activation(self, inputs):
        return K.layers.Activation("linear")(inputs)

    def _conv_kxk(self, inputs, name=None):
        return K.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
            kernel_regularizer=K.regularizers.L1L2(l1=1e-5, l2=1e-4),
            use_bias=False,
            name=name,
        )(inputs)

    def _conv_1x1(self, inputs, name=None):
        return K.layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
            kernel_regularizer=K.regularizers.L1L2(l1=1e-5, l2=1e-4),
            use_bias=False,
            name=name,
        )(inputs)

    def _dw_sep_conv_kxk(self, inputs, name=None):
        return K.layers.SeparableConv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
            kernel_regularizer=K.regularizers.L1L2(l1=1e-5, l2=1e-4),
            use_bias=False,
            name=name,
        )(inputs)

    def _dw_conv_kxk(self, inputs, name=None):
        return K.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
            kernel_regularizer=K.regularizers.L1L2(l1=1e-5, l2=1e-4),
            use_bias=False,
            name=name,
        )(inputs)


class VGGBlock(Block):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs

        x = self._conv_kxk(x)
        x = self._batch_norm(x, is_train)
        x = self._relu_activation(x)

        x = self._conv_kxk(x)
        x = self._batch_norm(x, is_train)
        x = self._relu_activation(x)

        return x


class ResNetBlock(Block):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs

        if(inputs.shape[-1] == self.filters):
            s = inputs
        else:
            # this is done to match filters in the shortcut
            s = self._conv_1x1(inputs)

        x = self._conv_kxk(x)
        x = self._batch_norm(x, is_train)
        x = self._relu_activation(x)

        x = self._conv_kxk(x)
        x = self._batch_norm(x, is_train)

        x = self._add([x, s])

        x = self._relu_activation(x)

        return x


class XceptionBlock(Block):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs

        if(inputs.shape[-1] == self.filters):
            s = inputs
        else:
            # this is done to match filters in the shortcut
            s = self._conv_1x1(inputs)

        x = self._dw_sep_conv_kxk(x)
        x = self._batch_norm(x, is_train)
        x = self._relu_activation(x)

        x = self._dw_sep_conv_kxk(x)
        x = self._batch_norm(x, is_train)

        x = self._add([x, s])

        x = self._relu_activation(x)

        return x


class MBConvBlock(Block):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs

        if(inputs.shape[-1] == self.filters):
            s = inputs
        else:
            # this is done to match filters in the shortcut
            s = self._conv_1x1(inputs)

        x = self._conv_1x1(x)
        x = self._batch_norm(x, is_train)
        x = self._relu6_activation(x)

        x = self._dw_conv_kxk(x)
        x = self._batch_norm(x, is_train)
        x = self._relu6_activation(x)

        x = self._conv_1x1(x)
        x = self._batch_norm(x, is_train)
        x = self._linear_activation(x)

        x = self._add([x, s])

        return x


class Identity(Block):
    def __call__(self, inputs, name=None, is_train=True):
        return inputs
