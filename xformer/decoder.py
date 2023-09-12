from tensorflow.keras.layers import Dropout, Layer, Input
from tensorflow.keras import Model
from xformer.common import AddAndNorm, FeedForward
from xformer.multihead_attention import MultiHeadAttention
from xformer.positional_encoding import CustomEmbeddingWithFixedPosnWts

class DecoderLayer(Layer):
    def __init__(self, sequence_length, n_heads, d_model, d_ff, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.build(input_shape=[None, sequence_length, d_model])
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.multihead_attention1 = MultiHeadAttention(n_heads, d_model)
        self.dropout1 = Dropout(dropout_rate)
        self.add_norm1 = AddAndNorm()
        self.multihead_attention2 = MultiHeadAttention(n_heads, d_model)
        self.dropout2 = Dropout(dropout_rate)
        self.add_norm2 = AddAndNorm()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout3 = Dropout(dropout_rate)
        self.add_norm3 = AddAndNorm()

    def call(self, x, mask, encoder_output, encoder_mask, training):
        # Multi-head self-attention layer
        multihead_output1 = self.multihead_attention1(x, x, x, mask)
        # Expected output shape = (batch_size, sequence_length, d_model)
        # Add in a dropout layer
        multihead_output1 = self.dropout1(multihead_output1, training=training)
        # Followed by an Add & Norm layer
        addnorm_output1 = self.add_norm1(x, multihead_output1)
        # Expected output shape = (batch_size, sequence_length, d_model)
        # Followed by another multi-head (cross-)attention layer
        multihead_output2 = self.multihead_attention2(
            addnorm_output1, encoder_output, encoder_output, encoder_mask
        )
        # Add in another dropout layer
        multihead_output2 = self.dropout2(multihead_output2, training=training)
        # Followed by another Add & Norm layer
        addnorm_output2 = self.add_norm2(addnorm_output1, multihead_output2)
        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output2)
        # Expected output shape = (batch_size, sequence_length, d_model)
        # Add in another dropout layer
        feedforward_output = self.dropout3(
            feedforward_output, training=training
        )
        # Followed by another Add & Norm layer
        return self.add_norm3(addnorm_output2, feedforward_output)
    
    def build_graph(self):
        input_layer = Input(shape=(self.sequence_length, self.d_model))
        return Model(inputs=[input_layer], outputs=self.call(input_layer, None, input_layer, None, True))
    
class Decoder(Layer):
    def __init__(
        self,
        vocab_size,
        sequence_length,
        n_heads,
        d_model,
        d_ff,
        n,
        dropout_rate,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pos_encoding = CustomEmbeddingWithFixedPosnWts(
            sequence_length, vocab_size, d_model
        )
        self.dropout = Dropout(dropout_rate)
        self.decoder_layer = [
            DecoderLayer(sequence_length, n_heads, d_model, d_ff, dropout_rate) for _ in range(n)
        ]

    def call(
        self,
        target_sequence,
        mask,
        encoder_output,
        encoder_mask,
        training,
    ):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(target_sequence)
        # Expected output shape = (number of sentences, sequence_length, d_model)
        # Add in a dropout layer
        x = self.dropout(pos_encoding_output, training=training)
        # Pass on the positional encoded values to each encoder layer
        for i, layer in enumerate(self.decoder_layer):
            x = layer(x, mask, encoder_output, encoder_mask, training)
        return x