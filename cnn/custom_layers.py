from keras.layers import Layer, Dense, Softmax
import tensorflow as tf

class SelfAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(SelfAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.units = input_shape[-1]
        self.query_dense = Dense(units=self.units)
        self.key_dense = Dense(units=self.units)
        self.value_dense = Dense(units=self.units)
        self.softmax = Softmax(axis=-1)
        super(SelfAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Reshape the input to (batch_size, height*width, channels)
        batch_size, height, width, channels = inputs.shape
        reshaped_inputs = tf.reshape(inputs, (batch_size, height * width, channels))

        query = self.query_dense(reshaped_inputs)
        key = self.key_dense(reshaped_inputs)
        value = self.value_dense(reshaped_inputs)

        attention_weights = tf.matmul(query, key, transpose_b=True)
        attention_weights = self.softmax(attention_weights)

        output = tf.matmul(attention_weights, value)
        output = tf.reshape(output, (batch_size, height, width, channels))
        return output