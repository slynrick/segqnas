from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    DepthwiseConv2D,
    ReLU,
    SeparableConv2D,
)


class Block(object):
    def __init__(self, kernel_size, filters, name=None):
        self.kernel_size = kernel_size
        self.filters = filters
        self.initializer = HeNormal(seed=0)
        self.padding = "same"
        self.data_format = "channels_last"

    def _add(self, inputs, name=None):
        return Add(name=name)(inputs)

    def _batch_norm(self, inputs, is_train, name=None):
        return BatchNormalization(axis=-1, momentum=0.9, epsilon=2e-5, name=name)(
            inputs=inputs, training=is_train
        )

    def _relu_activation(self, inputs):
        return Activation("relu")(inputs)

    def _relu6_activation(self, inputs):
        return ReLU(max_value=6.0)(inputs)

    def _linear_activation(self, inputs):
        return Activation("linear")(inputs)

    def _sigmoid_activation(self, inputs):
        return Activation("sigmoid")(inputs)

    def _conv_kxk(self, inputs, name=None):
        return Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
            use_bias=False,
            name=name,
        )(inputs)

    def _conv_1x1(self, inputs, name=None):
        return Conv2D(
            filters=self.filters,
            kernel_size=1,
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
            use_bias=False,
            name=name,
        )(inputs)

    def _dw_sep_conv_kxk(self, inputs, name=None):
        return SeparableConv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
            use_bias=False,
            name=name,
        )(inputs)

    def _dw_conv_kxk(self, inputs, name=None):
        return DepthwiseConv2D(
            kernel_size=self.kernel_size,
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
            use_bias=False,
            name=name,
        )(inputs)


class StemConvolution(Block):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs
        x = self._conv_kxk(x, name=f"{name}_conv")
        x = self._batch_norm(x, is_train=is_train, name=f"{name}_norm")
        x = self._relu_activation(x)
        return x


class OutputConvolution(Block):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs
        x = self._conv_1x1(x, name=f"{name}_conv")
        x = self._sigmoid_activation(x)
        return x


class VGGBlock(Block):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs

        x = self._conv_kxk(x, name=name + "_conv_1")
        x = self._batch_norm(x, is_train=is_train, name=name + "_norm_1")
        x = self._relu_activation(x)

        x = self._conv_kxk(x, name=name + "_conv_2")
        x = self._batch_norm(x, is_train=is_train, name=name + "_norm_2")
        x = self._relu_activation(x)

        return x


class ResNetBlock(Block):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs

        if inputs.shape[-1] == self.filters:
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

        if inputs.shape[-1] == self.filters:
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

        if inputs.shape[-1] == self.filters:
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
