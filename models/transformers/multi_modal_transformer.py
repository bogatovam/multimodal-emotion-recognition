from base.base_model import BaseModel
import tensorflow as tf

from models.layers.coattention_encoder_block import CoAttentionEncoderBlock
from models.layers.fusion.concatenation_fusion_layer import ConcatenationFusionLayer
from models.layers.fusion.fbr_fusion_layer import FactorizedPoolingFusionLayer
from models.layers.fusion.sum_fusion_layer import SumFusionLayer
import tensorflow_addons as tfa

from models.layers.multimodal_encoder_block import MultimodalEncoderBlock, MultimodalEncoderLayer
from models.layers.soft_attention import SoftAttention


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
                 pooling_size=-1,
                 fusion_type='sum',
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
        self._pooling_size = pooling_size

        self.model = self._build_model(co_attention, fusion_type, training)
        # self.model.summary()

        super(MultiModelTransformerModel, self).__init__(cp_dir=cp_dir,
                                                         cp_name=cp_name,
                                                         save_freq=log_and_save_freq_batch,
                                                         model=self.model,
                                                         iter_per_epoch=iter_per_epoch)

    def _build_model(self, co_attention, fusion_type, training):
        if not co_attention and fusion_type == 'sum':
            return self._build_sum_fusion_model(training=training)
        if not co_attention and fusion_type == 'concatenation':
            return self._build_concatenation_fusion_model(training=training)
        elif not co_attention and fusion_type == 'fbp':
            return self._build_fbp_fusion_model(training=training)
        elif not co_attention and fusion_type == 'mha':
            return self._build_mha_fusion_model(training=training)
        elif co_attention and fusion_type is 'sum':
            return self._build_co_attention_sum_fusion_model(training=training)
        elif co_attention and fusion_type is 'concatenation':
            return self._build_co_attention_concatenation_fusion_model(training=training)
        elif co_attention and fusion_type is 'fbp':
            return self._build_co_attention_fbp_fusion_model(training=training)
        elif co_attention and fusion_type is 'mha':
            return self._build_co_attention_mha_fusion_model(training=training)

    def _build_sum_fusion_model(self, training=True) -> tf.keras.Model:
        inputs = self._build_multimodal_input()
        intra_modality_outputs, attention_intra_modality_weights = self._build_usual_transformer_block(inputs, training)

        fusion_output = SumFusionLayer()(intra_modality_outputs)
        output_tensor = self._build_classification_layer(fusion_output)

        train_model = tf.keras.Model(inputs=inputs, outputs=output_tensor)
        test_model = tf.keras.Model(inputs=inputs, outputs=[output_tensor, *attention_intra_modality_weights])
        return train_model if training else test_model

    def _build_concatenation_fusion_model(self, training=True) -> tf.keras.Model:
        inputs = self._build_multimodal_input()
        intra_modality_outputs, attention_intra_modality_weights = self._build_usual_transformer_block(inputs, training)

        fusion_output = ConcatenationFusionLayer()(intra_modality_outputs)
        print(f'fusion_output.shape:={fusion_output.shape}')
        output_tensor = self._build_classification_layer(fusion_output)

        train_model = tf.keras.Model(inputs=inputs, outputs=output_tensor)
        test_model = tf.keras.Model(inputs=inputs, outputs=[output_tensor, *attention_intra_modality_weights])
        return train_model if training else test_model

    def _build_fbp_fusion_model(self, training=True) -> tf.keras.Model:
        inputs = self._build_multimodal_input()
        intra_modality_outputs, attention_intra_modality_weights = self._build_usual_transformer_block(inputs, training)

        fusion_output = FactorizedPoolingFusionLayer(self._dropout_rate, self._pooling_size)(intra_modality_outputs)
        output_tensor = self._build_classification_layer(fusion_output)

        train_model = tf.keras.Model(inputs=inputs, outputs=output_tensor)
        test_model = tf.keras.Model(inputs=inputs, outputs=[output_tensor, *attention_intra_modality_weights])
        return train_model if training else test_model

    def _build_mha_fusion_model(self, training=True) -> tf.keras.Model:
        inputs = self._build_multimodal_input()
        intra_modality_outputs, attention_intra_modality_weights = self._build_usual_transformer_block(inputs, training)

        fusion_output, _ = MultimodalEncoderLayer(d_model=self._d_model,
                                                  num_heads=self._num_heads,
                                                  dropout_rate=self._dropout_rate,
                                                  intermediate_fc_units_count=self._intermediate_fc_units_count,
                                                  soft_attention_output_units=1)(intra_modality_outputs, training, None)
        flatten_output = tf.keras.layers.Flatten()(fusion_output)
        output_tensor = self._build_classification_layer(flatten_output)
        train_model = tf.keras.Model(inputs=inputs, outputs=output_tensor)
        test_model = tf.keras.Model(inputs=inputs, outputs=[output_tensor, *attention_intra_modality_weights])
        return train_model if training else test_model

    def _build_co_attention_sum_fusion_model(self, training=True) -> tf.keras.Model:
        inputs = self._build_multimodal_input()
        intra_modality_outputs, attention_weights = self._build_co_attention_transformer_block(inputs, training)

        fusion_output = SumFusionLayer()(intra_modality_outputs)
        output_tensor = self._build_classification_layer(fusion_output)

        train_model = tf.keras.Model(inputs=inputs, outputs=output_tensor)
        test_model = tf.keras.Model(inputs=inputs, outputs=[output_tensor, attention_weights])
        return train_model if training else test_model

    def _build_co_attention_concatenation_fusion_model(self, training=True) -> tf.keras.Model:
        inputs = self._build_multimodal_input()
        intra_modality_outputs, attention_weights = self._build_co_attention_transformer_block(inputs, training)

        fusion_output = ConcatenationFusionLayer()(intra_modality_outputs)
        print(f'fusion_output.shape:={fusion_output.shape}')
        output_tensor = self._build_classification_layer(fusion_output)

        train_model = tf.keras.Model(inputs=inputs, outputs=output_tensor)
        test_model = tf.keras.Model(inputs=inputs, outputs=[output_tensor, attention_weights])
        return train_model if training else test_model

    def _build_co_attention_fbp_fusion_model(self, training=True) -> tf.keras.Model:
        inputs = self._build_multimodal_input()
        intra_modality_outputs, attention_weights = self._build_co_attention_transformer_block(inputs, training)

        fusion_output = FactorizedPoolingFusionLayer(self._dropout_rate, self._pooling_size)(intra_modality_outputs)
        output_tensor = self._build_classification_layer(fusion_output)

        train_model = tf.keras.Model(inputs=inputs, outputs=output_tensor)
        test_model = tf.keras.Model(inputs=inputs, outputs=[output_tensor, attention_weights])
        return train_model if training else test_model

    def _build_co_attention_mha_fusion_model(self, training=True) -> tf.keras.Model:
        inputs = self._build_multimodal_input()
        intra_modality_outputs, attention_weights = self._build_co_attention_transformer_block(inputs, training)

        fusion_output, _ = MultimodalEncoderLayer(d_model=self._d_model,
                                                  num_heads=self._num_heads,
                                                  dropout_rate=self._dropout_rate,
                                                  intermediate_fc_units_count=self._intermediate_fc_units_count,
                                                  soft_attention_output_units=1)(intra_modality_outputs,
                                                                                 training, None)
        flatten_output = tf.keras.layers.Flatten()(fusion_output)
        output_tensor = self._build_classification_layer(flatten_output)
        train_model = tf.keras.Model(inputs=inputs, outputs=output_tensor)
        test_model = tf.keras.Model(inputs=inputs, outputs=[output_tensor, attention_weights])
        return train_model if training else test_model

    def _build_multimodal_input(self):
        return [tf.keras.layers.Input(shape=modality.config.shape) for modality in self._modalities_list]

    def _build_usual_transformer_block(self, inputs, training):
        attention_intra_modality_weights = []

        print(f'input_0.shape:={inputs[0].shape}')
        intra_modality_outputs, weights = self._build_unimodal_transformer(inputs[0], training)
        attention_intra_modality_weights.append(weights)
        print(f'block{0}_output.shape:={intra_modality_outputs.shape}')

        for i in range(1, len(inputs)):
            print(f'input_{i}.shape:={inputs[i].shape}')
            output, weights = self._build_unimodal_transformer(inputs[i], training)

            intra_modality_outputs = tf.concat([intra_modality_outputs, output], axis=1)
            attention_intra_modality_weights.append(weights)
            print(f'block{i}_output.shape:={intra_modality_outputs.shape}')

        print(f'transformers output shape:={intra_modality_outputs.shape}')
        return intra_modality_outputs, attention_intra_modality_weights

    def _build_co_attention_transformer_block(self, inputs, training):
        attention_intra_modality_weights = []

        print(f'input_0.shape:={inputs[0].shape}')
        intra_modality_outputs, weights = self._build_unimodal_transformer(inputs[0], training)
        attention_intra_modality_weights.append(weights)
        print(f'block{0}_output.shape:={intra_modality_outputs.shape}')

        for i in range(1, len(inputs)):
            print(f'input_{i}.shape:={inputs[i].shape}')
            output, weights = self._build_co_attention_transformer(inputs[i - 1],
                                                                   inputs[i],
                                                                   training)

            intra_modality_outputs = tf.concat([intra_modality_outputs, output], axis=1)
            attention_intra_modality_weights.append(weights)
            print(f'block{i}_output.shape:={intra_modality_outputs.shape}')

        print(f'transformers output shape:={intra_modality_outputs.shape}')
        return intra_modality_outputs, attention_intra_modality_weights

    def _build_unimodal_transformer(self, input_, training):
        encoders_block = MultimodalEncoderBlock(num_layers=self._num_layers,
                                                d_model=self._d_model,
                                                num_heads=self._num_heads,
                                                dropout_rate=self._dropout_rate,
                                                max_features_count=self._max_features_count,
                                                intermediate_fc_units_count=self._intermediate_fc_units_count,
                                                soft_attention_output_units=input_.shape[1])
        encoder_output, attention_weights = encoders_block(input_, training=training)
        block_output = SoftAttention(self._intermediate_fc_units_count, self._dropout_rate)(encoder_output)
        # flatten_output = tf.keras.layers.Flatten()(block_output)
        return block_output, attention_weights

    def _build_co_attention_transformer(self, input_prev, input_curr, training):
        encoders_block = CoAttentionEncoderBlock(num_layers=self._num_layers,
                                                 d_model=self._d_model,
                                                 num_heads=self._num_heads,
                                                 dropout_rate=self._dropout_rate,
                                                 max_features_count=self._max_features_count,
                                                 intermediate_fc_units_count=self._intermediate_fc_units_count,
                                                 soft_attention_output_units=input_prev.shape[1])
        encoder_output, attention_weights = encoders_block(input_prev,
                                                           input_curr,
                                                           training=training)
        block_output = SoftAttention(self._intermediate_fc_units_count, self._dropout_rate)(encoder_output)
        # flatten_output = tf.keras.layers.Flatten()(block_output)
        return block_output, attention_weights

    def _build_classification_layer(self, features):
        return tf.keras.layers.Dense(units=self._num_classes, activation='sigmoid')(features)

    def get_train_model(self):
        lr = CustomSchedule(self._d_model)
        metrics = ['accuracy',
                   tfa.metrics.F1Score(name='f1_micro', num_classes=self._num_classes, average='micro'),
                   tfa.metrics.F1Score(name='f1_macro', num_classes=self._num_classes, average='macro')]
        self.model.compile(
            optimizer=self._optimizer(learning_rate=lr, beta_1=0.9,
                                      beta_2=0.98, epsilon=1e-9),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics
        )
        return self.model


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
