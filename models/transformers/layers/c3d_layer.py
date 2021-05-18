import tensorflow as tf


class C3dLayer(tf.keras.layers.Layer):
    def __init__(self, model):
        super(C3dLayer, self).__init__()
        self.model = model

    def build(self, input_shape):
        print("C3d input")
        print(input_shape)

    def call(self, inputs):
        print(inputs.shape)
        res = tf.map_fn(lambda batch_elem: self.model(batch_elem), inputs)
        print(res.shape)
        return res
