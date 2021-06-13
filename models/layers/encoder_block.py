import tensorflow as tf
import numpy as np

from models.layers.multi_head_attention import MultiHeadAttention


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, rate)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # Первый этап – вычислить матрицы запроса, ключа и значения.
        # Это делается с помощью формирования из эмбеддингов матрицы X и
        # ее умножения на матрицы весов, которые мы обучили (WQ, WK, WV)
        attn_output, attention_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attention_weights

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, intermediate_fc_units_count,
                 max_features_count,
                 dropout_rate=0.1):
        super(EncoderBlock, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.projection = tf.keras.layers.Dense(units=d_model)
        self.pos_encoding = self.positional_encoding(max_features_count,
                                                     self.d_model)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, intermediate_fc_units_count, dropout_rate)
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask=None):
        # (batch_size, input_seq_len, features_vector_len)
        seq_len = tf.shape(x)[1]

        attention_weights = {}
        # adding projection and position encoding.
        x = self.projection(x)  # (batch_size, input_seq_len - None?, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, attention = self.enc_layers[i](x, training, mask)
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
