import tensorflow as tf


class RPlus1Layer(tf.keras.layers.Layer):
    def __init__(self, model):
        super(RPlus1Layer, self).__init__()
        self.model = model.signatures["serving_default"]

    def build(self, input_shape):
        print("C3d input")
        print(input_shape)

    def call(self, inputs):
        res = tf.map_fn(lambda batch_elem: self._get_features_from_c3d(batch_elem), inputs)
        return res

    def _get_features_from_c3d(self, batch_elem):
        print(batch_elem.shape)
        return tf.map_fn(lambda el: self.model(tf.expand_dims(el, 0))['635'], batch_elem)
