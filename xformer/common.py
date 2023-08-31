from tensorflow.keras.layers import (
    Dense,
    Layer,
    LayerNormalization,
    ReLU,
)

class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super().__init__(**kwargs)
        self.fully_connected_1 = Dense(d_ff)  # First fully-connected layer
        self.fully_connected_2 = Dense(d_model)  # Second fully-connected layer
        self.activation = ReLU()  # ReLU activation layer to come in between

    def call(self, x):
        # The input is passed into the two fully-connected layers, with a ReLU in between
        fc1_output = self.fully_connected_1(x)
        fc2_output = self.fully_connected_2(self.activation(fc1_output))
        return fc2_output

class AddAndNorm(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = LayerNormalization()  # Layer normalization layer

    def call(self, x, sublayer_x):
        # Note: The sublayer's input and output need to be of the same shape to be summable
        add = x + sublayer_x
        # Apply layer normalization to the sum
        return self.layer_norm(add)