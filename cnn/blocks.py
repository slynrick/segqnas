from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    DepthwiseConv2D,
    ReLU,
    SeparableConv2D,
    concatenate,
)
from tensorflow.keras.regularizers import L2


class Block(object):
    def __init__(self, kernel_size, filters, name=None):
        self.data_format = "channels_last"
        self.filters = filters
        self.initializer = HeNormal(seed=0)
        self.kernel_size = kernel_size
        self.padding = "same"
        self.regularizer = L2(1e-6)

    def _add(self, inputs, name=None):
        return Add(name=f"{name}_Addition")(inputs)

    def _concat(self, inputs, name=None):
        return concatenate(inputs, name=f"{name}_Concatenation", axis=-1)

    def _avg_pooling(self, inputs, name=None):
        return AveragePooling2D(
            pool_size=2, stride=1, padding=self.padding, name=f"{name}_Pooling"
        )(inputs)

    def _batch_norm(self, inputs, is_train, name=None):
        return BatchNormalization(
            axis=-1, momentum=0.9, epsilon=2e-5, name=f"{name}_Normalization"
        )(inputs=inputs, training=is_train)

    def _relu_activation(self, inputs, name=None):
        return Activation("relu", name=f"{name}_ReLU")(inputs)

    # def _relu6_activation(self, inputs, name=None):
    #     return ReLU(max_value=6.0, name=f"{name}_ReLU6")(inputs)

    # def _linear_activation(self, inputs, name=None):
    #     return Activation("linear", name=f"{name}_Linear")(inputs)

    def _sigmoid_activation(self, inputs, name=None):
        return Activation("sigmoid", name=f"{name}_Sigmoid")(inputs)

    def _conv_kxk(self, inputs, name=None):
        return Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            use_bias=False,
            name=f"{name}_Convolution_{self.kernel_size}x{self.kernel_size}",
        )(inputs)

    def _conv_kx1(self, inputs, name=None):
        return Conv2D(
            filters=self.filters,
            kernel_size=(self.kernel_size, 1),
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            use_bias=False,
            name=f"{name}_Convolution_{self.kernel_size}x1",
        )(inputs)

    def _conv_1xk(self, inputs, name=None):
        return Conv2D(
            filters=self.filters,
            kernel_size=(1, self.kernel_size),
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            use_bias=False,
            name=f"{name}_Convolution_1x{self.kernel_size}",
        )(inputs)

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
            name=f"{name}_Convolution_1x1",
        )(inputs)

    # def _dw_sep_conv_kxk(self, inputs, name=None):
    #     return SeparableConv2D(
    #         filters=self.filters,
    #         kernel_size=self.kernel_size,
    #         activation=None,
    #         padding=self.padding,
    #         data_format=self.data_format,
    #         kernel_initializer=self.initializer,
    #         kernel_regularizer=self.regularizer,
    #         use_bias=False,
    #         name=name,
    #     )(inputs)

    # def _dw_conv_kxk(self, inputs, name=None):
    #     return DepthwiseConv2D(
    #         kernel_size=self.kernel_size,
    #         activation=None,
    #         padding=self.padding,
    #         data_format=self.data_format,
    #         kernel_initializer=self.initializer,
    #         kernel_regularizer=self.regularizer,
    #         use_bias=False,
    #         name=name,
    #     )(inputs)


class StemConvolution(Block):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs
        x = self._conv_kxk(x, name=f"{name}_Convolution")
        x = self._batch_norm(x, is_train=is_train, name=f"{name}_Normalization")
        x = self._relu_activation(x, name=f"{name}_Activation")
        return x


class OutputConvolution(Block):
    def _final_conv(self, inputs, name=None):
        return Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=None,
            padding=self.padding,
            data_format=self.data_format,
            kernel_initializer=self.initializer,
            kernel_regularizer=self.regularizer,
            bias_initializer=self.initializer,
            bias_regularizer=self.regularizer,
            name=name,
        )(inputs)

    def __call__(self, inputs, name=None, is_train=True):
        x = inputs
        x = self._final_conv(x, name=f"{name}_Convolution")
        x = self._sigmoid_activation(x, name=f"{name}_Activation")
        return x


class VGGBlock(Block):
    def __call__(self, inputs, name, is_train=True):
        x = inputs

        x = self._conv_kxk(x, name=f"{name}_1")
        x = self._batch_norm(x, is_train=is_train, name=f"{name}_1")
        x = self._relu_activation(x, name=f"{name}_1")

        x = self._conv_kxk(x, name=f"{name}_2")
        x = self._batch_norm(x, is_train=is_train, name=f"{name}_2")
        x = self._relu_activation(x, name=f"{name}_2")

        return x


class ResNetBlock(Block):
    def __call__(self, inputs, name, is_train=True):
        x = inputs

        if inputs.shape[-1] == self.filters:
            s = inputs
        else:
            # this is done to match filters in the shortcut
            s = self._conv_1x1(inputs, name=f"{name}_Shortcut")

        x = self._conv_kxk(x, name=f"{name}_1")
        x = self._batch_norm(x, is_train, name=f"{name}_1")
        x = self._relu_activation(x, name=f"{name}_1")

        x = self._conv_kxk(x, name=f"{name}_Convolution_2")
        x = self._batch_norm(x, is_train, name=f"{name}_2")

        x = self._add([x, s], name=f"{name}_Addition")

        x = self._relu_activation(x, name=f"{name}_2")

        return x


class DenseBlock(Block):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs

        s_0 = x

        x = self._batch_norm(x, is_train, name=f"{name}_1")
        x = self._relu_activation(x, name=f"{name}_1")
        x = self._conv_1x1(x, name=f"{name}_1")

        x = self._batch_norm(x, is_train, name=f"{name}_1")
        x = self._relu_activation(x, name=f"{name}_1")
        x = self._conv_kxk(x, name=f"{name}_1")

        s_1 = x

        x = self._concat([x, s_0], name=f"{name}_1")

        x = self._batch_norm(x, is_train, name=f"{name}_2")
        x = self._relu_activation(x, name=f"{name}_2")
        x = self._conv_1x1(x, name=f"{name}_2")

        x = self._batch_norm(x, is_train, name=f"{name}_2")
        x = self._relu_activation(x, name=f"{name}_2")
        x = self._conv_kxk(x, name=f"{name}_2")

        x = self._concat([x, s_0, s_1], name=f"{name}_2")

        return x


class InceptionBlock(Block):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs

        a = self._conv_1x1(x, name=f"{name}_Branch_1")

        b = self._avg_pooling(x, name=f"{name}_Branch_2")
        b = self._conv_1x1(b, name=f"{name}_Branch_2")

        c = self._conv_1x1(x, name=f"{name}_Branch_3")
        c = self._conv_kxk(c, name=f"{name}_Branch_3")
        d = self._conv_1xk(c, name=f"{name}_Branch_3")
        e = self._conv_kx1(c, name=f"{name}_Branch_3")

        x = self._concat([a, b, d, e], name=name)

        x = self._batch_norm(x, is_train, name=name)
        x = self._relu_activation(x, name=name)

        return x


# class XceptionBlock(Block):
#     def __call__(self, inputs, name=None, is_train=True):
#         x = inputs

#         if inputs.shape[-1] == self.filters:
#             s = inputs
#         else:
#             # this is done to match filters in the shortcut
#             s = self._conv_1x1(inputs)

#         x = self._dw_sep_conv_kxk(x)
#         x = self._batch_norm(x, is_train)
#         x = self._relu_activation(x)

#         x = self._dw_sep_conv_kxk(x)
#         x = self._batch_norm(x, is_train)

#         x = self._add([x, s])

#         x = self._relu_activation(x)

#         return x


# class MBConvBlock(Block):
#     def __call__(self, inputs, name=None, is_train=True):
#         x = inputs

#         if inputs.shape[-1] == self.filters:
#             s = inputs
#         else:
#             # this is done to match filters in the shortcut
#             s = self._conv_1x1(inputs)

#         x = self._conv_1x1(x)
#         x = self._batch_norm(x, is_train)
#         x = self._relu6_activation(x)

#         x = self._dw_conv_kxk(x)
#         x = self._batch_norm(x, is_train)
#         x = self._relu6_activation(x)

#         x = self._conv_1x1(x)
#         x = self._batch_norm(x, is_train)
#         x = self._linear_activation(x)

#         x = self._add([x, s])

#         return x


class IdentityBlock(Block):
    def __call__(self, inputs, name=None, is_train=True):
        return inputs
