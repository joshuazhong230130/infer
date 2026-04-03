import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        temperatures = temperatures.unsqueeze(dim=1)
        logits = logits.float().div_(temperatures)
        # equivalent to Gumbel-Max
        noise = torch.empty_like(logits).exponential_().clamp_min_(1e-10)
        sample_tokens = torch.argmax(logits - noise.log(), dim=-1, keepdim=True)
        return sample_tokens
