import os
from pathlib import Path
import torch
from torch import nn

from layers.attention import attention_with_kvcache, varlen_attention
from qwen3_config import Qwen3Config
from layers.layernorm import RMSNorm
from layers.position_embedding import get_rope
from layers.sampler import Sampler
from engine.kv_cache import KVCache

class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_size)
        self.out_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=config.dtype)

    def forward(self, input_ids, positions, is_prefill=True):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, positions, is_prefill=is_prefill)
        hidden_states = self.final_norm(hidden_states)
        if is_prefill:
            x = hidden_states[:, -1:, :].contiguous() # hidden state of last token of each seq
        else:
            x = hidden_states
        logits = self.out_head(x.to(self.config.dtype))
        return logits


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.self_attn = GroupedQueryAttention(
            d_in=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            num_kv_groups=config.num_key_value_heads,
            qk_norm=config.qk_norm,
            dtype=config.dtype,
            rope_theta=config.rope_base,
            max_position=config.context_length,
        )
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
    
    def forward(self, hidden_states, positions, is_prefill=True):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, positions, is_prefill)
        hidden_states = hidden_states + residual
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        
        return hidden_states

# Feed Forward
class Qwen3MLP(nn.Module):
    def __init__(self, config:Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, dtype=config.dtype, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, dtype=config.dtype, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, dtype=config.dtype, bias=False)
        self.act_fn = nn.functional.silu

    def forward(self, x):
        x_gate = self.gate_proj(x)
        x_up = self.up_proj(x)
        x = self.act_fn(x_gate) * x_up
        return self.down_proj(x)

class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_in,
        num_heads,
        num_kv_groups,
        head_dim=None,
        qk_norm=False,
        dtype=None,
        rope_theta: float = 10000,
        max_position: int = 4096 * 32,
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        
        if head_dim is None:
            assert d_in % num_heads == 0, "d_in must be divisible by num_heads if head_dim is not set"
            head_dim = d_in // num_heads
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        
        self.q_proj = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        
        self.o_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        
        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)
        else:
            self.q_norm = self.k_norm = None

        # KV Cache
        # self.register_buffer("K_cache", torch.zeros(1, self.num_kv_groups, 256, self.head_dim))
        # self.register_buffer("V_cache", torch.zeros(1, self.num_kv_groups, 256, self.head_dim))
        # self.cache_index = 0  # 当前已存 token 数

        self.rotary_emb = get_rope(
            self.head_dim,
            base=rope_theta,
            max_position=max_position,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.attn_cache = KVCache(1, self.num_kv_groups, 512, self.head_dim, device, dtype=dtype)

    def prefill_step(self, hidden_states, positions):
        x = hidden_states.squeeze(0)
        num_tokens = x.shape[0]

        # Apply projections
        query_states = self.q_proj(x) # [total_tokens, num_heads * head_dim]
        key_states = self.k_proj(x) # [total_tokens, num_kv_groups * head_dim]
        value_states = self.v_proj(x) # [total_tokens, num_kv_groups * head_dim]
        
        # Reshape
        query_states = query_states.view(-1, self.num_heads, self.head_dim)
        key_states = key_states.view(-1, self.num_kv_groups, self.head_dim)
        value_states = value_states.view(-1, self.num_kv_groups, self.head_dim)

        # Optional norm
        if self.q_norm:
            query_states = self.q_norm(query_states)
        if self.k_norm:
            key_states = self.k_norm(key_states)

        # Apply rope
        query_states, key_states = self.rotary_emb(positions, query_states, key_states)

        # append K/V to cache
        k_cache = key_states.unsqueeze(0)
        v_cache = value_states.unsqueeze(0)
        self.attn_cache.store_kvcache(k_cache, v_cache)

        # Expand K and V to match number of heads
        key_states = key_states.repeat_interleave(self.group_size, dim=1)
        value_states = value_states.repeat_interleave(self.group_size, dim=1)
        # Attention
        cu_seqlens = torch.tensor([0, num_tokens], dtype=torch.int32, device=hidden_states.device)
        max_seqlen = num_tokens
        context = varlen_attention(query_states, key_states, value_states, cu_seqlens, max_seqlen, scale=self.head_dim**(-0.5))
        context = context.reshape(num_tokens, self.d_out)
        out = self.o_proj(context)
        return out

    def decode_step(self, hidden_states, positions):
        b, num_tokens, _ = hidden_states.shape

        assert num_tokens == 1  # only one token forever
        x = hidden_states.squeeze(1)

        # Apply projections
        query_states = self.q_proj(x) # [b, 1, num_heads * head_dim]
        key_states = self.k_proj(x) # [b, 1, num_kv_groups * head_dim]
        value_states = self.v_proj(x) # [b, 1, num_kv_groups * head_dim]
        
        # Reshape
        query_states = query_states.view(b, self.num_heads, self.head_dim) # [b, num_heads, head_dim]
        key_states = key_states.view(b, self.num_kv_groups, self.head_dim)  # [b, num_kv_groups, head_dim]
        value_states = value_states.view(b, self.num_kv_groups, self.head_dim)  # [b, num_kv_groups, head_dim]

        # Optional norm
        if self.q_norm:
            query_states = self.q_norm(query_states)
        if self.k_norm:
            key_states = self.k_norm(key_states)

        # Apply rope for query, key
        query_states, key_states = self.rotary_emb(positions, query_states, key_states)

        # append K/V to cache
        k_cache, v_cache = self.attn_cache.store_kvcache(key_states.unsqueeze(1), value_states.unsqueeze(1))

        # Expand K and V to match number of heads
        k_cache = k_cache.repeat_interleave(self.group_size, dim=2)  # [batch_size, T, num_heads, head_dim]
        v_cache = v_cache.repeat_interleave(self.group_size, dim=2)  # [batch_size, T, num_heads, head_dim]

        # Attention
        cache_seq_lens = torch.tensor([k_cache.shape[1]], dtype=torch.int32, device=hidden_states.device)
        context = attention_with_kvcache(query_states, k_cache, v_cache, cache_seq_lens=cache_seq_lens, scale=self.head_dim**(-0.5))
        context = context.reshape(num_tokens, self.d_out)
        out = self.o_proj(context)
        return out

    def forward(self, hidden_states, positions, is_prefill=True):
        if is_prefill:
            return self.prefill_step(hidden_states, positions)
        else:
            return self.decode_step(hidden_states, positions)
