import importlib

from tensorflow import keras as K

from cnn.blocks import Block


class Cell(object):
    def __init__(self, block, kernel_size, filters):
        self.block = self._instatiate_block(block, kernel_size, filters)
        self.kernel_size = kernel_size
        self.filters = filters

    def _instatiate_block(self, block, kernel_size, filters):
        module = importlib.import_module("cnn.blocks")
        class_ = getattr(module, block)
        return class_(kernel_size, filters)

    def _concat(self, inputs, name=None):
        return K.layers.concatenate(inputs, name=name, axis=-1)

    def _conv_1x1(self, inputs, name=None):
        return K.layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            activation="relu",
            padding="same",
            data_format="channels_last",
            kernel_initializer="he_normal",
            use_bias=False,
            name=name,
        )(inputs)

    def _max_pool(self, inputs, name=None):
        return K.layers.MaxPooling2D(pool_size=(2, 2), name=name)(inputs)

    def _transpose_conv(self, inputs, name=None):
        return K.layers.Conv2DTranspose(
            filters=self.filters,
            kernel_size=3,
            strides=2,
            activation="relu",
            padding="same",
            data_format="channels_last",
            kernel_initializer="he_normal",
            use_bias=False,
            name=name,
        )(inputs)

    def _downsampling_conv(self, inputs, name=None):
        return K.layers.Conv2D(
            filters=self.filters,
            kernel_size=3,
            activation="relu",
            strides=2,
            padding="same",
            data_format="channels_last",
            kernel_initializer="he_normal",
            use_bias=False,
            name=name,
        )(inputs)


class DownscalingCell(Cell):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs

        if isinstance(inputs, list):
            x = self._concat(inputs)
            x = self._conv_1x1(x)

        x = self.block(x, is_train=is_train)
        x = self._downsampling_conv(x)
        # x = self._max_pool(x)

        return x


class UpscalingCell(Cell):
    def __call__(self, inputs, is_train=True):
        x = inputs

        if isinstance(inputs, list):
            x = self._concat(inputs)
            x = self._conv_1x1(x)

        x = self.block(x, is_train=is_train)
        x = self._transpose_conv(x)

        return x


class NonscalingCell(Cell):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs

        if isinstance(inputs, list):
            x = self._concat(inputs)
            x = self._conv_1x1(x)

        x = self.block(x, is_train=is_train)

        return x
