# model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BondInfluenceSelfAttention(nn.Module):
    """
    Multi-head attention where the attention scores are modulated by a bond inflence matrix.
    """
    def __init__(self, d_model, n_heads):
        super(BondInfluenceSelfAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear layers for Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self,
                x: torch.Tensor,
                bond_influence: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of shape [batch, seq_len, d_model] (projected features)
            bond_influence (torch.Tensor): Tensor of shape [batch, seq_len, seq_len] (bond influence matrix) 
        """
        batch_size, seq_len, _ = x.size()

        # Linear projections and reshape for multi-head attention.
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [B, n_heads, seq_len, d_k]
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [B, n_heads, seq_len, d_k]
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [B, n_heads, seq_len, d_k]

        # Scaled dot-product attention.
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, n_heads, seq_len, seq_len]
        attn_scores = attn_scores / math.sqrt(self.d_k)

        # Incorporate the bond influence matrix
        # Expand bond_influence to match attention heads
        bond_influence_expanded = bond_influence.unsqueeze(1)  # [B, 1, seq_len, seq_len]
        attn_scores = attn_scores * bond_influence_expanded  # TODO

        # Softmax over the last dimension
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, n_heads, seq_len, seq_len]

        # Weighted sum of the values.
        attn_output = torch.matmul(attn_weights, V)  # [B, n_heads, seq_len, d_k]

        # Combine heads.
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)  # [B, seq_len, d_model]
        return output
    
class BondInfluenceTransformerBlock(nn.Module):
    """
    A single transformer block that uses BondInfluenceAttention and a feed-forward network.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(BondInfluenceTransformerBlock, self).__init__()
        self.attn = BondInfluenceSelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                bond_influence: torch.Tensor):
        # Self-attention with bond influence modulation
        attn_out = self.attn(x, bond_influence)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Feed-forward network.
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        return x
    
class OCNTransformer(nn.Module):
    """
    A Perceiver-style model that takes as input two modalities:
        1. Atom type features: shape [batch, num_atoms, 3]
        2. Transformed spatial locations: shape [batch, num_atoms, 3]
    and integrates bond influence (shape [batch, num_atoms, num_atoms]) in the attention.
    The final output is a prediction of magnetic moment for each atom.
    """
    def __init__(self,
                 d_model=64,
                 n_heads=4,
                 num_layers=2,
                 dropout=0.1):
        super(OCNTransformer, self).__init__()
        input_dim = 6  #  3 (atom type) + 3 (spatial location)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.transformer_blocks = nn.ModuleList([
            BondInfluenceTransformerBlock(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])

        # Predict a scalar target (magnetic moment) per atom.
        self.out_layer = nn.Linear(d_model, 1)

    def forward(self,
                atom_type: torch.Tensor,
                spatial_location: torch.Tensor,
                bond_influence: torch.Tensor):
        """
        Args:
            atom_type (torch.Tensor): [batch, num_atoms, 3]
            spatial_location (torch.Tensor): [batch, num_atoms, 3]
            bond_influence (torch.Tensor): [batch, num_atoms, num_atoms]
        """
        # Concatenate the two modalities.
        x = torch.cat([atom_type, spatial_location], dim=-1)  # [B, num_atoms, 6]
        x = self.input_projection(x)  # [B, num_atoms, d_model]

        # Apply Transformer blocks.
        for block in self.transformer_blocks:
            x = block(x, bond_influence)

        # Predict per-atom target.
        out = self.out_layer(x)  # [B, num_atoms, 1]
        return out.squeeze(-1)  # [B, num_atoms]