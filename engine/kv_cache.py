import torch
from torch import nn

class KVCache:
    def __init__(self, batch_size, num_kv_heads, max_len, head_dim, device, dtype):
        self.k_cache = torch.zeros((batch_size, num_kv_heads, max_len, head_dim), device=device, dtype=dtype)  # [batch_size, num_kv_heads, max_seq_len, head_dim]
        self.v_cache = torch.zeros((batch_size, num_kv_heads, max_len, head_dim), device=device, dtype=dtype)  # [batch_size, num_kv_heads, max_seq_len, head_dim]
        self.current_idx = 0
        self.max_len = max_len

    def get_kv_for_attention(self, current_k, current_v):
        if self.current_idx == 0:
            return current_k, current_v
        else:
            past_k = self.k_cache[:, :, : self.current_idx, :]
            past_v = self.v_cache[:, :, : self.current_idx, :]
            attn_k = torch.cat((past_k, current_k), dim=2)
            attn_v = torch.cat((past_v, current_v), dim=2)
            return attn_k, attn_v

    def update_cache(self, k, v):
        assert self.current_idx < self.max_len
        self.k_cache[:, :, self.current_idx : self.current_idx + 1, :] = k
        self.v_cache[:, :, self.current_idx : self.current_idx + 1, :] = v
        self.current_idx += 1

    def prefill_kv(self, k, v):
        # k, v shape : [batch_size, num_kv_heads, tokens, head_dim]
        prefill_len = k.shape[2] # tokens
        assert prefill_len <= self.max_len
        self.k_cache[:, :, :prefill_len, :] = k
        self.v_cache[:, :, :prefill_len, :] = v
        self.current_idx = prefill_len
