# src/models/continuous_token_representation.py
import torch
import torch.nn as nn

class ContinuousTokenRepresentation(nn.Module):
    def __init__(self, vocab_size, embed_dim, dropout_rate=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.continuous_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )
    
    def forward(self, x):
        embedded = self.embed(x)
        return self.continuous_proj(embedded)

