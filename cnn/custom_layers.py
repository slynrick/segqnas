from keras.layers import Layer, Conv2D, Activation
import tensorflow as tf


class SelfAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)
        self.channels = -1
        self.query = None
        self.key = None
        self.value = None
        self.gamma = None

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.query = Conv2D(filters=self.channels, kernel_size=1, strides=1, padding='same')
        self.key = Conv2D(filters=self.channels, kernel_size=1, strides=1, padding='same')
        self.value = Conv2D(filters=self.channels, kernel_size=1, strides=1, padding='same')
        self.gamma = self.add_weight(shape=(self.channels,),
                                  initializer='glorot_normal',
                                  regularizer=tf.keras.regularizers.l2(1e-4))
        super(SelfAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)

        attention = tf.matmul(query, key, transpose_b=True) 
        attention = Activation('softmax')(attention)

        out = tf.matmul(attention, value)

        out = self.gamma * out + inputs
        return out