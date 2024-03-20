from keras.layers import Layer, Conv2D, Activation, Multiply
import tensorflow as tf


class SelfAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.channels = -1
        self.filters_f = None
        self.filters_g = None
        self.filters_h = None

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.filters_f = Conv2D(filters=self.channels, kernel_size=1, strides=1, padding='same')
        self.filters_g = Conv2D(filters=self.channels, kernel_size=1, strides=1, padding='same')
        self.filters_h = Conv2D(filters=self.channels, kernel_size=1, strides=1, padding='same')
        super(SelfAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        f = self.filters_f(inputs)
        g = self.filters_g(inputs)
        h = self.filters_h(inputs)

        s = tf.matmul(g, f, transpose_b=True) 
        beta = Activation('softmax')(s)

        o = tf.matmul(beta, h)

        x = Multiply()([o, inputs])
        return x