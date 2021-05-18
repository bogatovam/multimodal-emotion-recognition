import tensorflow as tf


class C3dLayer(tf.keras.layers.Layer):
    def __init__(self, model):
        super(C3dLayer, self).__init__()
        self.model = model

    def build(self, input_shape):
        print("C3d input")
        print(input_shape)

    def call(self, inputs):
        res = tf.map_fn(lambda batch_elem: self._get_features_from_c3d(batch_elem), inputs)
        return res

    def _get_features_from_c3d(self, batch_elem):
        for layer in self.model.layers:
            batch_elem = layer(batch_elem)
            if layer.name == "dropout_1":
                break
        print(batch_elem.shape)
        return batch_elem
