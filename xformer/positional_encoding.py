import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Layer, TextVectorization

class CustomEmbeddingWithFixedPosnWts(Layer):
    def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.word_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=output_dim
        )
        posn_embedding_matrix = self.get_position_encoding(
            sequence_length, output_dim
        )
        self.posn_embedding_layer = Embedding(
            input_dim=sequence_length,
            output_dim=output_dim,
            weights=[posn_embedding_matrix],
            trainable=False,
        )

    def get_position_encoding(self, seq_len, d, n=10000):
        pos_enc = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d / 2)):
                denominator = np.power(n, 2 * i / d)
                pos_enc[k, 2 * i] = np.sin(k / denominator)
                pos_enc[k, 2 * i + 1] = np.cos(k / denominator)
        return pos_enc

    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.posn_embedding_layer(position_indices)
        return embedded_words + embedded_indices