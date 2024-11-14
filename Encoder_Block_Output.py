import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.head_dim = embed_size // num_heads
        
        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by num_heads"
        
        self.query_linear = nn.Linear(embed_size, embed_size)
        self.key_linear = nn.Linear(embed_size, embed_size)
        self.value_linear = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, value, key, query, mask):
        batch_size = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]
        
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)
        
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, value_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        energy = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** (1/2))
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(energy, dim=-1)
        
        out = torch.matmul(attention, V)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, query_len, self.num_heads * self.head_dim)
        out = self.fc_out(out)
        
        return out

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AddNormLayer(nn.Module):
    def __init__(self, embed_size):
        super(AddNormLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(embed_size)
        
    def forward(self, input_tensor, output_tensor):
        residual = input_tensor + output_tensor
        return self.layer_norm(residual)

class EncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadSelfAttention(embed_size, num_heads)
        self.add_norm1 = AddNormLayer(embed_size)
        self.ffn = FeedForwardNetwork(embed_size, ff_hidden_size)
        self.add_norm2 = AddNormLayer(embed_size)
        
    def forward(self, value, key, query, mask=None):
        attention_output = self.mha(value, key, query, mask)
        attention_output = self.add_norm1(query, attention_output)
        
        ffn_output = self.ffn(attention_output)
        output = self.add_norm2(attention_output, ffn_output)
        
        return output

embed_size = 6
num_heads = 2
ff_hidden_size = 24
batch_size = 1
seq_len = 3

I = torch.tensor([[0.91, 1.84, 2.30, 1.52, 0.75, 0.12],
                  [0.68, 1.83, 1.10, 0.60, 1.40, 0.22],
                  [1.52, 1.01, 2.64, 1.15, 0.60, 0.30]])

mask = None

encoder_block = EncoderBlock(embed_size, num_heads, ff_hidden_size)

output = encoder_block(I, I, I, mask)

print("Output from Encoder Block:", output)
