from base.base_model import BaseModel
import tensorflow as tf
from models.transformers.layers.encoderblock import EncoderBlock
import tensorflow_addons as tfa

from models.transformers.layers.soft_attention import SoftAttention


class MultiModelTransformerModel(BaseModel):

    def __init__(self,
                 modalities_list,
                 num_layers,
                 d_model,
                 num_heads,
                 intermediate_fc_units_count,
                 num_classes,
                 max_features_count,
                 dropout_rate,
                 weight_decay,
                 cp_dir: str,
                 cp_name: str,
                 co_attention=False,
                 fusion_type='concatenation',
                 log_and_save_freq_batch: int = 100,
                 learning_rate: float = 0.001,
                 training: bool = True,
                 iter_per_epoch=390):
        self._modalities_list = modalities_list
        self._weight_decay = weight_decay
        self._learning_rate = learning_rate
        self._learning_rate = learning_rate

        self._optimizer = tfa.optimizers.AdamW

        self._num_layers = num_layers
        self._d_model = d_model
        self._num_heads = num_heads
        self._intermediate_fc_units_count = intermediate_fc_units_count
        self._num_classes = num_classes
        self._max_features_count = max_features_count
        self._dropout_rate = dropout_rate

        if not co_attention and fusion_type == 'concatenation':
            self.model = self._build_simple_concatenation_fusion_model(training=training)
        elif not co_attention and fusion_type == 'fbp':
            self.model = self._build_simple_fbp_fusion_model(training=training)
        elif co_attention and fusion_type is 'concatenation':
            self.model = self._build_co_attention_concatenation_fision_model(training=training)
        elif co_attention and fusion_type is 'fbp':
            self.model = self._build_co_attention_concatenation_fision_model(training=training)

        self.model.summary()

        super(MultiModelTransformerModel, self).__init__(cp_dir=cp_dir,
                                                         cp_name=cp_name,
                                                         save_freq=log_and_save_freq_batch,
                                                         model=self.model,
                                                         iter_per_epoch=iter_per_epoch)

    def _build_model(self, training=True) -> tf.keras.Model:
        inputs = [tf.keras.layers.Input(shape=modality.config.shape) for modality in self._modalities_list]

        self.encoders_block = EncoderBlock(num_layers=self._num_layers,
                                           d_model=self._d_model,
                                           num_heads=self._num_heads,
                                           dropout_rate=self._dropout_rate,
                                           max_features_count=self._max_features_count,
                                           intermediate_fc_units_count=self._intermediate_fc_units_count)

        encoder_output, attention_weights = self.encoders_block(input_tensor, training=training)
        block_output = SoftAttention(self._intermediate_fc_units_count, self._dropout_rate)(encoder_output)
        flatten_output = tf.keras.layers.Flatten()(block_output)
        output_tensor = tf.keras.layers.Dense(units=self._num_classes, activation='sigmoid')(flatten_output)
        train_model = tf.keras.Model(inputs=inputs, outputs=output_tensor)
        test_model = tf.keras.Model(inputs=inputs, outputs=[output_tensor, attention_weights])
        return train_model if training else test_model

    def get_train_model(self):
        metrics = ['accuracy',
                   tfa.metrics.F1Score(name='f1_micro', num_classes=self._num_classes, average='micro'),
                   tfa.metrics.F1Score(name='f1_macro', num_classes=self._num_classes, average='macro')]
        self.model.compile(
            optimizer=self._optimizer(learning_rate=self._learning_rate, weight_decay=self._weight_decay, beta_1=0.9,
                                      beta_2=0.98, epsilon=1e-9),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics
        )
        return self.model
