import tensorflow as tf


class FactorizedPoolingFusionLayer(tf.keras.layers.Layer):
    def __init__(self, dropout_rate, pooling_size):
        super(FactorizedPoolingFusionLayer, self).__init__()

        self._pooling_size = pooling_size
        self._dropout = tf.keras.layers.Dropout(dropout_rate)
        self._layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, **kwargs):
        fusion_result = tf.reduce_prod(inputs, axis=1)

        fusion_result = self._dropout(fusion_result)

        sum_pooling_result = tf.signal.frame(fusion_result, self._pooling_size, self._pooling_size, axis=1)

        sum_pooling_result = tf.math.reduce_sum(sum_pooling_result, axis=2)
        return self._layer_norm(sum_pooling_result)
