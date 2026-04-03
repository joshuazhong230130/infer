from functools import lru_cache
import torch
from torch import nn
from transformers import Qwen3Config


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float,
        max_position_embeddings: int,
    ) -> None:
        super().__init__()
        cos, sin = self.compute_rope_params(
            head_dim=head_dim,
            theta_base=base,
            max_position_embeddings=max_position_embeddings
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def compute_rope_params(self, head_dim, theta_base=10_000, max_position_embeddings=4096, dtype=torch.float32):
        assert head_dim % 2 == 0, "Embedding dimension must be even"
        
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
        positions = torch.arange(max_position_embeddings, dtype=dtype)
        angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0) # Shape: [max_position_embeddings, head_dim // 2]
        # Expand angles to match head dim
        angles = torch.cat([angles, angles], dim=1) # Shape: [max_position_embeddings, head_dim]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        return cos, sin

    def forward(self, x, offset=0):
        # x: [batch_size, num_heads, seq_len, head_dim]
        batch_size, num_heads, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even"
        
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2:]
        
        cos = self.cos[offset:offset+seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim // 2)
        sin = self.sin[offset:offset+seq_len, :].unsqueeze(0).unsqueeze(0)
        
        rotated = torch.cat((-x2, x1), dim=-1)
        x_rotated = (x * cos) + (rotated * sin)
        
        return x_rotated.to(dtype=x.dtype)

@lru_cache(1)
def get_rope(
    head_size: int,
    base: float,
    max_position: int,
):
    rotary_emb = RotaryEmbedding(head_size, base, max_position)
    return rotary_emb
