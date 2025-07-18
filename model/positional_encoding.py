"""Positional encoding implementations for Dendrite Transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation.
    
    Based on the paper: https://arxiv.org/abs/2104.09864
    Efficient for long sequences and extrapolates well.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 131072, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute the frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for positions
        self._cached_positions = None
        self._cached_freqs = None
        self._cached_seq_len = 0
    
    def _compute_cos_sin(self, seq_len: int, device: torch.device):
        """Compute cosine and sine for rotary embeddings."""
        if self._cached_seq_len >= seq_len and self._cached_freqs is not None:
            return self._cached_freqs[:seq_len]
        
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        
        # Create complex frequencies
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        
        # Cache the results
        self._cached_freqs = freqs_complex
        self._cached_seq_len = seq_len
        
        return freqs_complex
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embedding to query and key tensors."""
        seq_len = q.shape[-2]
        device = q.device
        
        # Get frequencies
        freqs_complex = self._compute_cos_sin(seq_len, device)
        
        # Reshape for broadcasting
        # q, k shape: [batch, heads, seq_len, head_dim]
        q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        
        # Apply rotation
        freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim//2]
        q_rotated = torch.view_as_real(q_complex * freqs_complex).flatten(-2)
        k_rotated = torch.view_as_real(k_complex * freqs_complex).flatten(-2)
        
        return q_rotated.type_as(q), k_rotated.type_as(k)


class ALiBiPositionalBias(nn.Module):
    """Attention with Linear Biases (ALiBi) implementation.
    
    Based on the paper: https://arxiv.org/abs/2108.12409
    No learned parameters, works well for any sequence length.
    """
    
    def __init__(self, num_heads: int, max_seq_len: int = 131072):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Compute slopes for each head
        slopes = self._compute_slopes(num_heads)
        self.register_buffer("slopes", slopes)
        
        # Cache for bias matrix
        self._cached_bias = None
        self._cached_seq_len = 0
    
    def _compute_slopes(self, num_heads: int) -> torch.Tensor:
        """Compute the slopes for ALiBi attention heads."""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            slopes = torch.tensor(get_slopes_power_of_2(num_heads))
        else:
            # Closest power of 2
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes_power_of_2 = get_slopes_power_of_2(closest_power_of_2)
            
            # Interpolate remaining slopes
            extra_slopes = []
            for i in range(1, num_heads - closest_power_of_2 + 1):
                extra_slopes.append(
                    slopes_power_of_2[0] * (2 ** (-(i - 1) / (num_heads - closest_power_of_2)))
                )
            
            slopes = torch.tensor(slopes_power_of_2 + extra_slopes)
        
        return slopes.view(1, num_heads, 1, 1)
    
    def _compute_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute the ALiBi bias matrix."""
        if self._cached_seq_len >= seq_len and self._cached_bias is not None:
            return self._cached_bias[:, :, :seq_len, :seq_len]
        
        # Create position indices
        positions = torch.arange(seq_len, device=device).view(1, 1, seq_len, 1)
        memory_positions = torch.arange(seq_len, device=device).view(1, 1, 1, seq_len)
        
        # Compute relative positions
        relative_positions = memory_positions - positions
        relative_positions = relative_positions.abs().neg().float()
        
        # Apply slopes
        alibi_bias = relative_positions * self.slopes.to(device)
        
        # Cache the result
        self._cached_bias = alibi_bias
        self._cached_seq_len = seq_len
        
        return alibi_bias
    
    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """Add ALiBi bias to attention scores.
        
        Args:
            attention_scores: Tensor of shape [batch, heads, seq_len, seq_len]
            
        Returns:
            Biased attention scores
        """
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        device = attention_scores.device
        
        # Get bias matrix
        bias = self._compute_bias(seq_len, device)
        
        # Add bias to attention scores
        return attention_scores + bias


class SinusoidalPositionEmbedding(nn.Module):
    """Standard sinusoidal position embedding.
    
    Used in the original Transformer paper.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 131072):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create position encoding matrix
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * 
            -(math.log(10000.0) / dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """Get positional encoding for given sequence length."""
        return self.pe[:, :seq_len]


class LearnedPositionEmbedding(nn.Module):
    """Learned position embedding (current implementation)."""
    
    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, dim)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Get positional embeddings for given positions."""
        return self.embedding(positions)


def create_position_encoding(
    encoding_type: str,
    dim: int,
    max_seq_len: int = 131072,
    num_heads: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """Factory function to create position encoding modules.
    
    Args:
        encoding_type: One of 'learned', 'sinusoidal', 'rope', 'alibi'
        dim: Dimension of the embeddings
        max_seq_len: Maximum sequence length
        num_heads: Number of attention heads (required for ALiBi)
        **kwargs: Additional arguments for specific encodings
        
    Returns:
        Position encoding module
    """
    if encoding_type == 'learned':
        return LearnedPositionEmbedding(max_seq_len, dim)
    elif encoding_type == 'sinusoidal':
        return SinusoidalPositionEmbedding(dim, max_seq_len)
    elif encoding_type == 'rope':
        return RotaryPositionEmbedding(dim, max_seq_len, **kwargs)
    elif encoding_type == 'alibi':
        if num_heads is None:
            raise ValueError("ALiBi requires num_heads parameter")
        return ALiBiPositionalBias(num_heads, max_seq_len)
    else:
        raise ValueError(f"Unknown position encoding type: {encoding_type}")