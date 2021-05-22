import tensorflow as tf
import numpy as np


class SumFusionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SumFusionLayer, self).__init__()

    def call(self, inputs, **kwargs):
        fusion_result = tf.reduce_sum(inputs, axis=1)
        return fusion_result
