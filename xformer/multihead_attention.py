from tensorflow import cast, float32, math, matmul, reshape, shape, transpose
from tensorflow.keras.backend import softmax
from tensorflow.keras.layers import Dense, Layer

class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, queries, keys, values, mask=None):
        d_k = shape(keys)[-1]
        # Score the queries against the keys after transposing the latter, and then scale
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(
            cast(d_k, float32)
        )
        # Apply mask to the attention scores
        if mask is not None:
            scores += float("-inf") * mask
        # Compute the weights using a softmax operation
        weights = softmax(scores)
        # Compute attention by a weighted sum of the value vectors
        return matmul(weights, values)

class MultiHeadAttention(Layer):
    def __init__(self, n_heads, d_model, **kwargs):
        super().__init__(**kwargs)

        assert d_model % h == 0
        d_k = d_model // h
        # We assume d_v always equals d_k
        d_v = d_k

        self.attention = DotProductAttention()  # Scaled dot product attention
        self.n_heads = n_heads  # Number of attention heads
        self.d_k = d_k  # Dim of the linearly projected queries and keys
        self.d_v = d_v  # Dim of the linearly projected values
        self.W_q = Dense(d_k)  # Learned projection matrix for the queries, ...
        self.W_k = Dense(d_k)  # ... for the keys
        self.W_v = Dense(d_v)  # ... for the values
        self.W_o = Dense(d_model)  # ... for the multi-head output

    def reshape_tensor(self, x, n_heads, do_split_flag):
        if do_split_flag:
            # Tensor shape after reshaping and transposing:
            # (batch_size, n_heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], n_heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations:
            # (batch_size, seq_length, d_model)
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], -1))
        return x

    def call(self, queries, keys, values, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.n_heads, True)
        # Resulting tensor shape: (batch_size, n_heads, input_seq_length, -1)

        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.n_heads, True)
        # Resulting tensor shape: (batch_size, n_heads, input_seq_length, -1)

        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values), self.n_heads, True)
        # Resulting tensor shape: (batch_size, n_heads, input_seq_length, -1)

        # Compute the multi-head attention output using the reshaped queries, keys,
        # and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, mask)
        # Resulting tensor shape: (batch_size, n_heads, input_seq_length, -1)

        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.n_heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)

        # Apply one final linear projection to the output to generate the multi-head
        # attention. Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.W_o(output)