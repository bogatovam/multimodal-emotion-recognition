import tensorflow as tf


class ConcatenationFusionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ConcatenationFusionLayer, self).__init__()

    def call(self, inputs, **kwargs):
        return tf.map_fn(lambda x: tf.squeeze(tf.reshape(x, shape=(1, 1024))), inputs)
