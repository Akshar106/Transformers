import numpy as np

word_embeddings = np.array([
    [0.25, 0.47, 0.69],
    [0.13, 0.53, 0.21],
    [0.44, 0.12, 0.85]
])

positional_encodings = np.array([
    [0.30, 0.35, 0.50],
    [0.18, 0.20, 0.55],
    [0.49, 0.52, 0.15]
])

encoded_embeddings = word_embeddings + positional_encodings
sequence_length, embedding_dim = encoded_embeddings.shape

np.random.seed(0) 
W_q = np.random.rand(embedding_dim, embedding_dim)
W_k = np.random.rand(embedding_dim, embedding_dim)
W_v = np.random.rand(embedding_dim, embedding_dim)

queries = np.dot(encoded_embeddings, W_q)
keys = np.dot(encoded_embeddings, W_k)
values = np.dot(encoded_embeddings, W_v)

def scaled_dot_product_attention(queries, keys, values):
    # Calculate the dot products of the queries and keys to get the raw attention scores
    scores = np.dot(queries, keys.T)
    
    # Scale by the square root of the embedding dimension for stability
    scale_factor = np.sqrt(embedding_dim)
    scaled_scores = scores / scale_factor
    
    # Apply softmax to get attention weights
    attention_weights = np.exp(scaled_scores) / np.sum(np.exp(scaled_scores), axis=1, keepdims=True)
    
    # Multiply the attention weights by the values
    attention_output = np.dot(attention_weights, values)
    
    return attention_output, attention_weights

attention_output, attention_weights = scaled_dot_product_attention(queries, keys, values)

print("Queries:\n", queries)
print("\nKeys:\n", keys)
print("\nValues:\n", values)
print("\nAttention Weights:\n", attention_weights)
print("\nAttention Output:\n", attention_output)
