import tensorflow as tf
import numpy as np

from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.soft_attention import SoftAttention


class CoAttentionEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, intermediate_fc_units_count, dropout_rate, soft_attention_output_units,
                 is_last):
        super(CoAttentionEncoderLayer, self).__init__()

        self.is_last = is_last
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, intermediate_fc_units_count)

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout4 = tf.keras.layers.Dropout(dropout_rate)

        self._soft_attention = SoftAttention(intermediate_fc_units_count, dropout_rate, soft_attention_output_units)

    def call(self, x, y, training, mask):
        # Первый этап – вычислить матрицы запроса, ключа и значения.
        # Это делается с помощью формирования из эмбеддингов матрицы X и
        # ее умножения на матрицы весов, которые мы обучили (WQ, WK, WV)
        attn_output, attention_weights = self.mha1(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        attn_output2, attention_weights2 = self.mha2(y, y, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output2 = self.dropout2(attn_output2, training=training)
        out2 = self.layer_norm2(out1 + attn_output2)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layer_norm3(out2 + ffn_output)  # (batch_size, input_seq_len, d_model)

        if self.is_last:
            return out3, attention_weights

        soft_att_output = self._soft_attention(out3)
        soft_att_output = self.dropout4(soft_att_output, training=training)
        out4 = self.layer_norm4(out3 + soft_att_output)  # (batch_size, input_seq_len, d_model)

        return out4, attention_weights

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])


class CoAttentionEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, intermediate_fc_units_count, max_features_count,
                 dropout_rate, soft_attention_output_units):
        super(CoAttentionEncoderBlock, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.projection = tf.keras.layers.Dense(units=d_model)
        self.pos_encoding = self.positional_encoding(max_features_count,
                                                     self.d_model)

        self.enc_layers = [CoAttentionEncoderLayer(d_model, num_heads, intermediate_fc_units_count, dropout_rate,
                                                   soft_attention_output_units, i == num_layers - 1)
                           for i in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, y, training, mask=None):
        # (batch_size, input_seq_len, features_vector_len)
        seq_len = tf.shape(x)[1]

        attention_weights = {}
        # adding projection and position encoding.
        x = self.projection(x)  # (batch_size, input_seq_len - None?, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, attention = self.enc_layers[i](x, y, training, mask)
            attention_weights[f'encoder_layer{i + 1}_block1'] = attention

        return x, attention_weights  # (batch_size, input_seq_len, d_model)

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
