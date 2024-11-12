import numpy as np

word_embeddings = np.array([
    [0.25, 0.47, 0.69],
    [0.13, 0.53, 0.21],
    [0.44, 0.12, 0.85]
])

sequence_length, embedding_dim = word_embeddings.shape

def get_positional_encoding(seq_len, d_model):
    positional_encoding = np.zeros((seq_len, d_model))
    
    # Calculate positional encodings
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            positional_encoding[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
            if i + 1 < d_model:
                positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** (2 * i / d_model)))
    return positional_encoding

positional_encodings = get_positional_encoding(sequence_length, embedding_dim)
encoded_embeddings = word_embeddings + positional_encodings

print("Word Embeddings:\n", word_embeddings)
print("\nPositional Encodings:\n", positional_encodings)
