import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None
    
    def forward(self, x):
        input_dtype = x.dtype
        
        if self.qwen3_compatible:
            x = x.to(torch.float32)
        
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale
        
        if self.shift is not None:
            norm_x = norm_x + self.shift
        
        return norm_x.to(input_dtype)
