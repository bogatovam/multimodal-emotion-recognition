import tensorflow as tf


class ConcatenationFusionLayer(tf.keras.layers.Layer):
    def __init__(self, dmodel, num_modalities):
        self.d_model = dmodel
        self.num_modalities = num_modalities
        super(ConcatenationFusionLayer, self).__init__()

    def call(self, inputs, **kwargs):
        return tf.map_fn(lambda x: tf.squeeze(tf.reshape(x, shape=(1, self.d_model * self.num_modalities))), inputs)
