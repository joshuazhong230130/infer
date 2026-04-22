import torch
from torch import nn

class KVCache:
    def __init__(self, batch_size, num_kv_heads, max_len, head_dim, device, dtype):
        self.k_cache = torch.zeros((batch_size, max_len, num_kv_heads, head_dim), device=device, dtype=dtype)  # [batch_size, max_seq_len, num_kv_heads, head_dim]
        self.v_cache = torch.zeros((batch_size, max_len, num_kv_heads, head_dim), device=device, dtype=dtype)  # [batch_size, max_seq_len, num_kv_heads, head_dim]
        self.current_idx = 0
        self.max_len = max_len

    def store_kvcache(self, k_cache, v_cache):
        # k_cache, v_cache shape : [batch_size, N, num_kv_heads, head_dim]
        N = k_cache.shape[1]
        assert N <= self.max_len
        start = self.current_idx
        end = self.current_idx + N
        self.k_cache[:, start : end, :, :] = k_cache
        self.v_cache[:, start : end, :, :] = v_cache
        self.current_idx = end
        k_caches = self.k_cache[:, : end, :, :]
        v_caches = self.v_cache[:, : end, :, :]
        return k_caches, v_caches
