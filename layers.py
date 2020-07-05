import tensorflow as tf
from tensorflow.keras.layers import *


# Difference b/n batch-norm and instance norm
# batch_norm => (batch_size, m, n, c) -> mean, var shape = (1, 1, 1, c)
# instance norm => (batch_size, m, n, c) -> mean, var shape = (batch_size, 1, 1, c)
# batch_size - no of samples
# m, n - spatial dimensions
# c - no of channels


# Custom Instance Normalization Layer
class InstanceNormalization(Layer):
    def __init__(self, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)

    # Use this method to do lazy calling because we don't know the input shape until we pass in the input to the layer
    def build(self, input_shape):
        channels = input_shape[3]
        self.scale = self.add_weight(name="channels" + str(channels),
                                     shape=[channels],
                                     initializer=tf.random_normal_initializer(1.0, 0.02)
                                     )
        self.shift = self.add_weight(name="shift_" + str(channels),
                                    shape=[channels],
                                    initializer=tf.random_normal_initializer(1.0, 0.02)
                                    )
    def call(self, inputs):
        epsilon = 1e-5
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (inputs - mean) * inv
        return self.scale * normalized + self.shift

