from base.base_model import BaseModel
import tensorflow as tf

from models.transformers.layers.c3d_layer import C3dLayer


class TransformerModel(BaseModel):

    def __init__(self,
                 extractor,
                 cp_dir: str,
                 cp_name: str,
                 pretrained_model_path: str = "",
                 num_fine_tuned_layers: int = 2,
                 log_and_save_freq_batch: int = 100,
                 learning_rate: float = 0.001,
                 trained: bool = False,
                 iter_per_epoch=390):
        self._extractor = extractor
        self._pretrained_feature_extractor = tf.keras.models.load_model(pretrained_model_path, compile=False)
        self._pretrained_feature_extractor.trainable = False

        self._num_fine_tuned_layers = num_fine_tuned_layers
        self._learning_rate = learning_rate

        self._optimizer = tf.keras.optimizers.Adam

        self._input_shape = (10, 16, 112, 112, 3)
        self._activation = 'relu'
        self._first_layer_num_neurons = 48
        self._second_layer_num_neurons = 34
        self._initializer = tf.keras.initializers.glorot_normal

        self.train_model, self.test_model = self._build_model()
        self.model = self.test_model if trained else self.train_model

        super(TransformerModel, self).__init__(cp_dir=cp_dir,
                                               cp_name=cp_name,
                                               save_freq=log_and_save_freq_batch,
                                               model=self.model,
                                               iter_per_epoch=iter_per_epoch)

    def _build_model(self):
        input_tensor = tf.keras.layers.Input(shape=self._input_shape)
        print(input_tensor.shape)
        x = C3dLayer(self._pretrained_feature_extractor)(input_tensor)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=self._first_layer_num_neurons, activation=self._activation)(x)
        # todo 9
        output_tensor = tf.keras.layers.Dense(units=9, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
        model.summary()
        return model, model

    def get_train_model(self):
        metrics = ['accuracy']
        self.train_model.compile(
            optimizer=self._optimizer(learning_rate=self._learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics
        )
        return self.train_model

    def get_test_model(self):
        return self.test_model
