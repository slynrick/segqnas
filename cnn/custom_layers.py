from keras.layers import Layer, Conv2D, Activation, Multiply
import tensorflow as tf


class SelfAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.filters = -1
        self.query = None
        self.key = None
        self.value = None

    def build(self, input_shape):
        self.filters = input_shape[-1]
        self.query = Conv2D(filters=self.filters, kernel_size=1, strides=1, padding='same')
        self.key = Conv2D(filters=self.filters, kernel_size=1, strides=1, padding='same')
        self.value = Conv2D(filters=self.filters, kernel_size=1, strides=1, padding='same')
        super(SelfAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)

        attention = tf.matmul(key, query, transpose_b=True) 
        attention = Activation('softmax')(attention)

        output = tf.matmul(attention, value)

        output = Multiply()([output, inputs])
        return output