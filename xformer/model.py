from tensorflow import cast, float32, linalg, math, maximum, newaxis, ones
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from xformer.decoder import Decoder
from xformer.encoder import Encoder

class Xformer(Model):
    def padding_mask(self, inuput):
        # Create mask marking zero padding values in the input by 1s
        mask = math.equal(input, 0)
        mask = cast(mask, float32)

        # The shape of the mask should be broadcastable to the shape
        # of the attention weights that it will be masking later on
        return mask[:, newaxis, newaxis, :]

    def lookahead_mask(self, n_tokens):
        # Mask out "future" entries by marking them with a 1.0
        mask = 1 - linalg.band_part(ones((n_tokens, n_tokens)), -1, 0)

        return mask

    def __init__(
        self,
        enc_vocab_size,
        dec_vocab_size,
        enc_seq_len,
        dec_seq_len,
        n_heads,
        d_model,
        d_ff_inner,
        n_layers,
        dropout_rate,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Set up the encoder block
        self.encoder = Encoder(
            enc_vocab_size,
            enc_seq_len,
            n_heads,
            d_model,
            d_ff_inner,
            n_layers,
            dropout_rate,
        )

        # Set up the decoder block
        self.decoder = Decoder(
            dec_vocab_size,
            dec_seq_len,
            n_heads,
            d_model,
            d_ff_inner,
            n_layers,
            dropout_rate,
        )

        # Define the final Dense layer that maps output probabilities to tokens in the target vocabulary
        self.final_layer = Dense(dec_vocab_size)

    def call(self, enc_input, dec_input, training):
        # Create padding mask to mask the encoder inputs as well as the encoder outputs (which are input to the decoder)
        enc_mask = self.padding_mask(enc_input)

        # Create and combine padding and look-ahead masks to be fed into the decoder
        dec_padding_mask = self.padding_mask(dec_input)
        dec_lookahead_mask = self.lookahead_mask(dec_input.shape[1])
        dec_mask = maximum(dec_padding_mask, dec_lookahead_mask)

        # Feed inputs into the encoder
        enc_output = self.encoder(enc_input, enc_padding_mask, training)

        # Feed encoder output into the decoder
        dec_output = self.decoder(
            dec_input, dec_mask, enc_output, enc_mask, training
        )

        # Pass decoder output through a final Dense layer
        final_output = self.final_layer(dec_output)

        return final_output