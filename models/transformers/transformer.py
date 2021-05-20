from base.base_model import BaseModel
import tensorflow as tf
from models.transformers.layers.encoderblock import EncoderBlock
import tensorflow_addons as tfa

from models.transformers.layers.soft_attention import SoftAttention


class TransformerModel(BaseModel):

    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 intermediate_fc_units_count,
                 num_classes,
                 max_features_count,
                 input_shape,
                 dropout_rate,
                 weight_decay,
                 cp_dir: str,
                 cp_name: str,
                 log_and_save_freq_batch: int = 100,
                 learning_rate: float = 0.001,
                 training: bool = True,
                 iter_per_epoch=390):
        self._weight_decay = weight_decay
        self._learning_rate = learning_rate

        self._optimizer = tfa.optimizers.AdamW

        self._input_shape = input_shape
        self._features_len = input_shape[0]
        self._num_layers = num_layers
        self._d_model = d_model
        self._num_heads = num_heads
        self._intermediate_fc_units_count = intermediate_fc_units_count
        self._num_classes = num_classes
        self._max_features_count = max_features_count
        self._dropout_rate = dropout_rate

        self.model = self._build_model(training=training)
        self.model.summary()

        super(TransformerModel, self).__init__(cp_dir=cp_dir,
                                               cp_name=cp_name,
                                               save_freq=log_and_save_freq_batch,
                                               model=self.model,
                                               iter_per_epoch=iter_per_epoch)

    def _build_model(self, training=True) -> tf.keras.Model:
        input_tensor = tf.keras.layers.Input(shape=self._input_shape)
        self.encoders_block = EncoderBlock(num_layers=self._num_layers,
                                           d_model=self._d_model,
                                           num_heads=self._num_heads,
                                           dropout_rate=self._dropout_rate,
                                           max_features_count=self._max_features_count,
                                           intermediate_fc_units_count=self._intermediate_fc_units_count)

        enc_padding_mask = self.create_padding_mask(input_tensor)
        # (batch_size, features_len, d_model)
        encoder_output, attention_weights = self.encoders_block(input_tensor, mask=enc_padding_mask, training=training)
        block_output = SoftAttention(self._intermediate_fc_units_count, self._dropout_rate)(encoder_output)
        flatten_output = tf.keras.layers.Flatten()(block_output)
        output_tensor = tf.keras.layers.Dense(units=self._num_classes, activation='sigmoid')(flatten_output)
        train_model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
        test_model = tf.keras.Model(inputs=input_tensor, outputs=[output_tensor, attention_weights])
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

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        print(f'seq.shape:={seq.shape}')
        return seq[:, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
