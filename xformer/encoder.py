from tensorflow.keras.layers import Dropout, Layer
from xformer.multihead_attention import MultiHeadAttention
from xformer.positional_encoding import CustomEmbeddingWithFixedPosnWts

class EncoderLayer(Layer):
    def __init__(self, n_heads, d_model, d_ff, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.multihead_attention = MultiHeadAttention(n_heads, d_model)
        self.dropout1 = Dropout(dropout_rate)
        self.add_norm1 = AddAndNorm()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(dropout_rate)
        self.add_norm2 = AddAndNorm()

    def call(self, x, padding_mask, training):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x, x, x, padding_mask)
        # Expected output shape = (batch_size, sequence_length, d_model)
        # Add in a dropout layer
        multihead_output = self.dropout1(multihead_output, training=training)
        # Followed by an Add & Norm layer
        addnorm_output1 = self.add_norm1(x, multihead_output)
        # Expected output shape = (batch_size, sequence_length, d_model)
        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output1)
        # Expected output shape = (batch_size, sequence_length, d_model)
        # Add in another dropout layer
        feedforward_output = self.dropout2(
            feedforward_output, training=training
        )
        # Followed by another Add & Norm layer
        return self.add_norm2(addnorm_output1, feedforward_output)

class Encoder(Layer):
    def __init__(
        self,
        vocab_size,
        sequence_length,
        n_heads,
        d_model,
        d_ff,
        n_enc_layers,
        dropout_rate,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.wrd_emb_posn_enc = CustomEmbeddingWithFixedPosnWts(
            sequence_length, vocab_size, d_model
        )
        self.dropout = Dropout(dropout_rate)
        self.encoder_layers = [
            EncoderLayer(n_heads, d_model, d_ff, dropout_rate)
            for _ in range(n_enc_layers)
        ]

    def call(self, input_sentence, padding_mask, training):
        # Generate the word embeddings & positional encodings
        emb_enc_output = self.wrd_emb_posn_enc(input_sentence)
        # Expected output shape = (batch_size, sequence_length, d_model)
        # Add in a dropout layer
        x = self.dropout(emb_enc_output, training=training)
        # Feed the result into the stack of encoder layers
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, padding_mask, training)
        return x