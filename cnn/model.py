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


def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)

def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes):
    model_input = Input(shape=(image_size, image_size, 3))
    resnet50 = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return Model(inputs=model_input, outputs=model_output)

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
            return layers.MaxPooling2D(
                pool_size=self.pool_size,
                strides=self.strides,
                data_format="channels_last",
                padding=self.padding,
                name=name,
            )(inputs)

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
            return layers.AveragePooling2D(
                pool_size=self.pool_size,
                strides=self.strides,
                data_format="channels_last",
                padding=self.padding,
                name=name,
            )(inputs)
        else:
            return inputs


class NoOp(object):
    pass


def get_segmentation_model(
    input_shape, num_classes, fn_dict, net_list, is_train=True, mu=0.9, epsilon=2e-5
):

    """layer_dict = {}
    for name, definition in fn_dict.items():
        if definition["function"] in ["ConvBlock"]:
            definition["params"]["mu"] = mu
            definition["params"]["epsilon"] = epsilon
        layer_dict[name] = globals()[definition["function"]](**definition["params"])

    skip_connections = []
    inputs = Input(shape=input_shape)
    x = inputs

    for i, f in enumerate(net_list):
        if f == "no_op":
            continue
        elif isinstance(layer_dict[f], ConvBlock):
            x = layer_dict[f](inputs=x, name=f"l{i}_{f}", is_train=is_train)
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
                size=(2, 2),
                data_format="channels_last",
                name=f"l{i+len(net_list)}_upsampling",
            )(x)
            x = layers.Concatenate()([x, skip_connections.pop()])

    outputs = layers.Conv2D(
        filters=num_classes,
        kernel_size=1,
        activation="sigmoid",
        padding="same",
        strides=1,
        data_format="channels_last",
        kernel_initializer="he_normal",
        bias_initializer="he_normal",
        name="final_conv",
    )(x)

    model = Model(inputs=inputs, outputs=outputs)"""
    inputs = Input(shape=input_shape)
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    # outputs
    logits = layers.Conv2D(num_classes, 1, padding="same")(u9)
    outputs = layers.Activation("softmax")(logits)

    # unet model with Keras Functional API
    model = Model(inputs, outputs)

    return model


def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = layers.Conv2D(
        n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal"
    )(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(
        n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal"
    )(x)
    return x


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x
