import tensorflow as tf
from tensorflow.keras import layers

class TransformerEncoderLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerEncoder(layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.layers_list = [
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, rate)
            for _ in range(num_layers)
        ]

    def call(self, inputs, training):
        x = inputs
        for layer in self.layers_list:
            x = layer(x, training=training)
        return x

class TransformerClassifier(tf.keras.Model):
    def __init__(self, vocab_size, max_len, num_layers, embed_dim, num_heads, ff_dim, num_classes, rate=0.1):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embed_dim, input_length=max_len)
        self.pos_encoding = self.positional_encoding(max_len, embed_dim)
        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, rate)
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dropout = layers.Dropout(rate)
        self.fc = layers.Dense(num_classes, activation="softmax")

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        angle_rads = tf.concat([tf.expand_dims(sines, -1), tf.expand_dims(cosines, -1)], axis=-1)
        angle_rads = tf.reshape(angle_rads, [position, d_model])
        return tf.cast(angle_rads[tf.newaxis, ...], dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        return pos / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x += self.pos_encoding[:, :tf.shape(x)[1], :]
        x = self.encoder(x, training=training)
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        return self.fc(x)
