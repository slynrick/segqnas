from keras.initializers import GlorotUniform, HeUniform
from keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    concatenate,
    DepthwiseConv2D,
    ReLU
)
from keras.regularizers import L2
import cnn.custom_layers as custom_layers


class Block(object):
    def __init__(self, kernel_size, filters, name=None):
        self.data_format = "channels_last"
        self.filters = filters
        self.initializer = HeUniform(seed=0)
        self.kernel_size = kernel_size
        self.padding = "same"
        self.regularizer = L2(1e-6)

    def _add(self, inputs, name=None):
        return Add(name=f"{name}_Addition")(inputs)

    def _concat(self, inputs, name=None):
        return concatenate(inputs, name=f"{name}_Concatenation", axis=-1)

    def _avg_pooling(self, inputs, name=None):
        return AveragePooling2D(
            pool_size=2, strides=1, padding=self.padding, name=f"{name}_Pooling"
        )(inputs)

    def _batch_norm(self, inputs, is_train, name=None):
        return BatchNormalization(
            axis=-1, momentum=0.9, epsilon=2e-5, name=f"{name}_Normalization"
        )(inputs=inputs, training=is_train)

    def _relu_activation(self, inputs, name=None):
        return Activation("relu", name=f"{name}_ReLU")(inputs)
    
    def _relu6_activation(self, inputs, name):
        return ReLU(6.0, name=f"{name}_ReLU6")(inputs)

    def _sigmoid_activation(self, inputs, name=None):
        return Activation("sigmoid", name=f"{name}_Sigmoid")(inputs)

    def _softmax_activation(self, inputs, name=None):
        return Activation("softmax", name=f"{name}_Softmax")(inputs)

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
    
    def _depthwise_conv(self, inputs, name):
        return DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False, name=f"{name}_DepthwiseConv3x3")(inputs)

    def _swish_activation(self, inputs, name):
        return inputs * Activation("sigmoid", name=f"{name}_Swish")(inputs)
    
    def _selfattention(self, inputs, name=None):
        return custom_layers.SelfAttentionLayer(
            name=f"{name}_SelfAttention",
        )(inputs)


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
            kernel_initializer=GlorotUniform(seed=0),
            kernel_regularizer=self.regularizer,
            bias_initializer=GlorotUniform(seed=0),
            bias_regularizer=self.regularizer,
            name=f"{name}_{self.kernel_size}x{self.kernel_size}",
        )(inputs)

    def __call__(self, inputs, name=None, is_train=True):
        x = inputs
        x = self._final_conv(x, name=f"{name}_Convolution")

        if self.kernel_size == 1:
            x = self._sigmoid_activation(x, name=f"{name}_Activation")
        else:
            x = self._softmax_activation(x, name=f"{name}_Activation")
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

        x = self._batch_norm(x, is_train, name=f"{name}_2")
        x = self._relu_activation(x, name=f"{name}_2")
        x = self._conv_kxk(x, name=f"{name}_1")

        s_1 = x

        x = self._concat([x, s_0], name=f"{name}_1")

        x = self._batch_norm(x, is_train, name=f"{name}_3")
        x = self._relu_activation(x, name=f"{name}_3")
        x = self._conv_1x1(x, name=f"{name}_2")

        x = self._batch_norm(x, is_train, name=f"{name}_4")
        x = self._relu_activation(x, name=f"{name}_4")
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

        x = self._concat([a, b, d, e], name=f"{name}_Block")

        x = self._batch_norm(x, is_train, name=f"{name}_Block")
        x = self._relu_activation(x, name=f"{name}_Block")

        return x


class IdentityBlock(Block):
    def __call__(self, inputs, name=None, is_train=True):
        return inputs
    

class SelfAttentionBlock(Block):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs
        x = self._selfattention(x, name=f"{name}_Block")

        return x

class MobileNetBlock(Block):
    def __call__(self, inputs, name, is_train=True):
        x = inputs

        # Depthwise Convolution
        x = self._depthwise_conv(x, name=f"{name}_dw")
        x = self._batch_norm(x, is_train=is_train, name=f"{name}_dw")
        x = self._relu_activation(x, name=f"{name}_dw")

        # Pointwise Convolution
        x = self._conv_1x1(x, name=f"{name}_pw")
        x = self._batch_norm(x, is_train=is_train, name=f"{name}_pw")
        x = self._relu_activation(x, name=f"{name}_pw")

        return x

class MobileNetV2Block(Block):
    def __call__(self, inputs, expansion_factor, out_channels, stride, name, is_train=True):
        x = inputs
        in_channels = x.shape[-1]

        # Expansion phase (Pointwise convolution)
        x = self._conv_1x1(x, filters=in_channels * expansion_factor, name=f"{name}_expand")
        x = self._batch_norm(x, is_train=is_train, name=f"{name}_expand")
        x = self._relu6_activation(x, name=f"{name}_expand")

        # Depthwise convolution
        x = self._depthwise_conv(x, stride=stride, name=f"{name}_dw")
        x = self._batch_norm(x, is_train=is_train, name=f"{name}_dw")
        x = self._relu6_activation(x, name=f"{name}_dw")

        # Linear bottleneck (Pointwise convolution without activation)
        x = self._conv_1x1(x, filters=out_channels, name=f"{name}_project")
        x = self._batch_norm(x, is_train=is_train, name=f"{name}_project")

        # Residual connection if input and output have the same shape
        if stride == 1 and in_channels == out_channels:
            x = Add(name=f"{name}_residual")([inputs, x])

        return x

class EfficientNetBlock(Block):
    def __call__(self, inputs, name, is_train=True):
        x = inputs

        # Depthwise separable convolution 3x3
        x = self._depthwise_conv(x, name=f"{name}_dw")
        x = self._batch_norm(x, is_train=is_train, name=f"{name}_dw")
        x = self._swish_activation(x, name=f"{name}_dw")

        # Pointwise convolution 1x1
        x = self._conv_1x1(x, name=f"{name}_pw")
        x = self._batch_norm(x, is_train=is_train, name=f"{name}_pw")
        x = self._swish_activation(x, name=f"{name}_pw")

        return x

class SqueezeNetBlock(Block):
    def __call__(self, inputs, name, is_train=True):
        x = inputs

        # Squeeze part (1x1 convolutions)
        squeeze_channels = int(x.shape[-1]) // 4
        x = self._conv_kxk(x, filters=squeeze_channels, kernel_size=(1, 1), name=f"{name}_squeeze")
        x = self._batch_norm(x, is_train=is_train, name=f"{name}_squeeze")
        x = self._relu_activation(x, name=f"{name}_squeeze")

        # Expand part (1x1 and 3x3 convolutions)
        x = self._conv_kxk(x, filters=squeeze_channels * 4, kernel_size=(1, 1), name=f"{name}_expand1")
        x = self._batch_norm(x, is_train=is_train, name=f"{name}_expand1")
        x = self._relu_activation(x, name=f"{name}_expand1")

        x = self._conv_kxk(x, filters=squeeze_channels * 4, kernel_size=(3, 3), name=f"{name}_expand2")
        x = self._batch_norm(x, is_train=is_train, name=f"{name}_expand2")
        x = self._relu_activation(x, name=f"{name}_expand2")

        return x

######################################### Self att concatenated blocks
class VGGBlockSelfAtt(Block):
    def __call__(self, inputs, name, is_train=True):
        x = inputs

        x = self._conv_kxk(x, name=f"{name}_1")
        x = self._batch_norm(x, is_train=is_train, name=f"{name}_1")
        x = self._relu_activation(x, name=f"{name}_1")

        x = self._conv_kxk(x, name=f"{name}_2")
        x = self._batch_norm(x, is_train=is_train, name=f"{name}_2")
        x = self._relu_activation(x, name=f"{name}_2")

        x = self._selfattention(x, name=f"{name}_Block")

        return x


class ResNetBlockSelfAtt(Block):
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

        x = self._selfattention(x, name=f"{name}_Block")

        return x


class DenseBlockSelfAtt(Block):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs

        s_0 = x

        x = self._batch_norm(x, is_train, name=f"{name}_1")
        x = self._relu_activation(x, name=f"{name}_1")
        x = self._conv_1x1(x, name=f"{name}_1")

        x = self._batch_norm(x, is_train, name=f"{name}_2")
        x = self._relu_activation(x, name=f"{name}_2")
        x = self._conv_kxk(x, name=f"{name}_1")

        s_1 = x

        x = self._concat([x, s_0], name=f"{name}_1")

        x = self._batch_norm(x, is_train, name=f"{name}_3")
        x = self._relu_activation(x, name=f"{name}_3")
        x = self._conv_1x1(x, name=f"{name}_2")

        x = self._batch_norm(x, is_train, name=f"{name}_4")
        x = self._relu_activation(x, name=f"{name}_4")
        x = self._conv_kxk(x, name=f"{name}_2")

        x = self._concat([x, s_0, s_1], name=f"{name}_2")

        x = self._selfattention(x, name=f"{name}_Block")

        return x


class InceptionBlockSelfAtt(Block):
    def __call__(self, inputs, name=None, is_train=True):
        x = inputs

        a = self._conv_1x1(x, name=f"{name}_Branch_1")

        b = self._avg_pooling(x, name=f"{name}_Branch_2")
        b = self._conv_1x1(b, name=f"{name}_Branch_2")

        c = self._conv_1x1(x, name=f"{name}_Branch_3")
        c = self._conv_kxk(c, name=f"{name}_Branch_3")
        d = self._conv_1xk(c, name=f"{name}_Branch_3")
        e = self._conv_kx1(c, name=f"{name}_Branch_3")

        x = self._concat([a, b, d, e], name=f"{name}_Block")

        x = self._batch_norm(x, is_train, name=f"{name}_Block")
        x = self._relu_activation(x, name=f"{name}_Block")

        x = self._selfattention(x, name=f"{name}_Block")

        return x