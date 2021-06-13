from kapre.time_frequency import Melspectrogram
from keras_vggface import VGGFace

from base.base_model import BaseModel
import tensorflow as tf

from configs.dataset.modality import DatasetFeaturesSet
from models.layers.fusion.concatenation_fusion_layer import ConcatenationFusionLayer
from models.layers.fusion.fbr_fusion_layer import FactorizedPoolingFusionLayer
from models.layers.fusion.sum_fusion_layer import SumFusionLayer


class MultimodalModel(BaseModel):

    def __init__(self,
                 modalities_list,
                 fc_units,
                 first_layer,
                 second_layer,
                 cp_dir: str,
                 cp_name: str,
                 fusion_type='sum',
                 dropout=0.1,
                 log_and_save_freq_batch: int = 100,
                 learning_rate: float = 0.001,
                 trained: bool = False):

        self._modalities_list = modalities_list
        self._learning_rate = learning_rate

        self._optimizer = tf.keras.optimizers.Adam

        self._lstm_units_first_layer = first_layer
        self._lstm_units_second_layer = second_layer
        self._activation = 'relu'
        self._fc_units = fc_units
        self._dropout = dropout
        self._pooling_size = 4
        self.train_model, self.test_model = self._build_model(fusion_type)
        self.model = self.test_model if trained else self.train_model

        super(MultimodalModel, self).__init__(cp_dir=cp_dir,
                                              cp_name=cp_name,
                                              save_freq=log_and_save_freq_batch,
                                              model=self.model)

    def _build_model(self, fusion_type):
        if fusion_type == 'sum':
            return self._build_sum_fusion_model()
        if fusion_type == 'concatenation':
            return self._build_concatenation_fusion_model()
        elif fusion_type == 'fbp':
            return self._build_fbp_fusion_model()

    def _build_sum_fusion_model(self):
        inputs = self._build_multimodal_input()
        intra_modality_features = []
        for i, modality in enumerate(self._modalities_list):
            if modality == DatasetFeaturesSet.PULSE:
                intra_modality_features.append(self.pulse_conv_net(inputs[i]))
            else:
                intra_modality_features.append(inputs[i])
        intra_modality_outputs = self._build_LSTM_block(intra_modality_features)
        fusion_output = SumFusionLayer()(intra_modality_outputs)
        output_tensor = self._build_classification_layer(fusion_output)

        model = tf.keras.Model(inputs=inputs, outputs=output_tensor)
        model.summary()
        return model, model

    def _build_concatenation_fusion_model(self):
        inputs = self._build_multimodal_input()
        intra_modality_features = []
        for i, modality in enumerate(self._modalities_list):
            if modality == DatasetFeaturesSet.PULSE:
                intra_modality_features.append(self.pulse_conv_net(inputs[i]))
            else:
                intra_modality_features.append(inputs[i])
        intra_modality_outputs = self._build_LSTM_block(intra_modality_features)
        fusion_output = ConcatenationFusionLayer(self._lstm_units_second_layer, len(self._modalities_list))(
            intra_modality_outputs)
        output_tensor = self._build_classification_layer(fusion_output)

        model = tf.keras.Model(inputs=inputs, outputs=output_tensor)
        model.summary()
        return model, model

    def _build_fbp_fusion_model(self):
        inputs = self._build_multimodal_input()
        intra_modality_features = []
        for i, modality in enumerate(self._modalities_list):
            if modality == DatasetFeaturesSet.PULSE:
                intra_modality_features.append(self.pulse_conv_net(inputs[i]))
            else:
                intra_modality_features.append(inputs[i])
        intra_modality_outputs = self._build_LSTM_block(intra_modality_features)
        fusion_output = FactorizedPoolingFusionLayer(self._dropout, self._pooling_size)(intra_modality_outputs)
        output_tensor = self._build_classification_layer(fusion_output)

        model = tf.keras.Model(inputs=inputs, outputs=output_tensor)
        model.summary()
        return model, model

    def get_train_model(self):
        metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                   tf.keras.metrics.AUC()]
        self.train_model.compile(
            optimizer=self._optimizer(learning_rate=self._learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics
        )
        return self.train_model

    def get_test_model(self):
        return self.test_model

    def apply_lstm(self, x):
        x = tf.keras.layers.LSTM(units=self._lstm_units_first_layer, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(units=self._lstm_units_second_layer)(x)
        return x

    def pulse_conv_net(self, input_tensor):
        x = tf.keras.layers.Conv2D(16,
                                   kernel_size=(input_tensor.shape[1] // 2, 2),
                                   strides=(2, 1),
                                   activation=self._activation)(input_tensor)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(x)
        x = tf.keras.layers.Reshape((x.shape[1], 16))(x)
        return x

    def _build_multimodal_input(self):
        return [tf.keras.layers.Input(shape=modality.config.input_shape) for modality in self._modalities_list]

    def _build_classification_layer(self, features):
        x = tf.keras.layers.Dense(units=self._fc_units, activation=self._activation)(features)
        return tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

    def _build_LSTM_block(self, inputs):
        outputs = []
        for i in range(0, len(inputs)):
            outputs.append(self.apply_lstm(inputs[i]))

        return tf.stack(outputs, axis=1)
