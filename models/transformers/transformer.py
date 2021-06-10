from base.base_model import BaseModel
import tensorflow as tf
from models.layers.encoderblock import EncoderBlock

from models.layers.soft_attention import SoftAttention
import tensorflow_addons as tfa


class F1Micro(tfa.metrics.F1Score):

    def __init__(self, threshold=0.2, **kwargs):
        super(F1Micro, self).__init__(name='f1_micro', threshold=threshold, num_classes=7, average='micro', **kwargs)
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_true = y_true > self.threshold
        super(F1Micro, self).update_state(y_true, y_pred)


class F1Macro(tfa.metrics.F1Score):

    def __init__(self, threshold=0.2, **kwargs):
        super(F1Macro, self).__init__(name='f1_macro', threshold=threshold, num_classes=7, average='macro', **kwargs)
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_true = y_true > self.threshold
        super(F1Macro, self).update_state(y_true, y_pred)


class Accuracy(tf.keras.metrics.Metric):

    def __init__(self, name='accuracy_threshold', threshold=0.2, **kwargs):
        super(Accuracy, self).__init__(name=name, **kwargs)

        self.threshold = threshold
        self.m = tf.keras.metrics.Accuracy()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast((y_true > self.threshold), tf.float32)
        y_pred = tf.cast((y_pred > self.threshold), tf.float32)

        self.m.update_state(y_true, y_pred)

    def result(self):
        return self.m.result()


class TransformerModel(BaseModel):

    def __init__(self,
                 activation,
                 optimizer,
                 regularizer,
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
                 soft_attention_output_units=1,
                 log_and_save_freq_batch: int = 100,
                 learning_rate: float = 0.001,
                 training: bool = True,
                 iter_per_epoch=390):
        self._weight_decay = weight_decay
        self._learning_rate = learning_rate
        self._regularizer = regularizer
        self._activation = activation

        self._optimizer = optimizer

        self._input_shape = input_shape
        self._features_len = input_shape[0]
        self._num_layers = num_layers
        self._d_model = d_model
        self._num_heads = num_heads
        self._soft_attention_output_units = soft_attention_output_units
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
        self.encoders_block = EncoderBlock(
            activation=self._activation,
            regularizer=self._regularizer,
            num_layers=self._num_layers,
            d_model=self._d_model,
            num_heads=self._num_heads,
            dropout_rate=self._dropout_rate,
            max_features_count=self._max_features_count,
            intermediate_fc_units_count=self._intermediate_fc_units_count)

        # enc_padding_mask = self.create_padding_mask(input_tensor)
        # (batch_size, features_len, d_model)
        encoder_output, attention_weights = self.encoders_block(input_tensor, training=training)
        block_output = SoftAttention(self._regularizer, self._activation, self._intermediate_fc_units_count,
                                     self._dropout_rate)(
            encoder_output)
        flatten_output = tf.keras.layers.Flatten()(block_output)
        output_tensor = tf.keras.layers.Dense(units=self._num_classes, activation='sigmoid')(flatten_output)
        train_model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
        test_model = tf.keras.Model(inputs=input_tensor, outputs=[output_tensor, attention_weights])
        return train_model if training else test_model

    def get_train_model(self):
        metrics = [Accuracy(threshold=0.4), F1Micro(threshold=0.4), F1Macro(threshold=0.4)]

        optimizer = None
        if self._optimizer.__name__ == 'Adam':
            optimizer = self._optimizer(learning_rate=self._learning_rate, beta_1=0.9,
                                        beta_2=0.98, epsilon=1e-9)
        else:
            optimizer = self._optimizer(learning_rate=self._learning_rate, beta_1=0.9,
                                        beta_2=0.98, epsilon=1e-9, weight_decay=self._weight_decay)

        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics
        )
        return self.model

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        print(f'seq.shape:={seq.shape}')
        return seq[:, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * self.warmup_steps ** -1.5

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
