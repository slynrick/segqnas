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

    def _concat(self, inputs, name=None):
        return K.layers.concatenate(inputs, name=name, axis=-1)

    def _batch_norm(self, inputs, is_train, name=None):
        return K.layers.BatchNormalization(
            axis=-1, momentum=0.9, epsilon=2e-5, name=name
        )(inputs=inputs, training=is_train)

    def _activation(self, inputs):
        return K.layers.Activation("relu")(inputs)

    def _conv_kxk(self, inputs, name=None):
        return K.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
            use_bias=False,
            name=name,
        )(inputs)

    def _conv_1xk(self, inputs, name=None):
        return K.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, self.kernel_size),
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
            use_bias=False,
            name=name,
        )(inputs)

    def _conv_kx1(self, inputs, name=None):
        return K.layers.Conv2D(
            filters=self.filters,
            kernel_size=(self.kernel_size, 1),
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
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
            use_bias=False,
            name=name,
        )(inputs)

    def _avg_pool(self, inputs, name=None):
        return K.layers.AveragePooling2D(pool_size=(2, 2))(inputs)


class VGGBlock(Block):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs

        x = self._conv_kxk(x)
        x = self._batch_norm(x, is_train)
        x = self._activation(x)

        x = self._conv_kxk(x)
        x = self._batch_norm(x, is_train)
        x = self._activation(x)

        return x


class ResNetBlock(Block):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs
        skip = self._conv_1x1(x)

        x = self._conv_kxk(x)
        x = self._batch_norm(x, is_train)
        x = self._activation(x)

        x = self._conv_kxk(x)
        x = self._batch_norm(x, is_train)

        x = self._add([x, skip])

        x = self._activation(x)

        return x


class InceptionBlock(Block):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs

        b1 = self._conv_1x1(x)

        b2 = self._avg_pool(x)
        b2 = self._conv_1x1(b2)

        b3 = self._conv_1x1(x)
        b3 = self._conv_kxk(b3)
        b3_1 = self._conv_1xk(b3)
        b3_2 = self._conv_kx1(b3)

        x = self._concat([b1, b2, b3_1, b3_2])
        x = self._batch_norm(x, is_train)
        x = self._activation(x)

        return x
