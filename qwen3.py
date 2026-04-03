import os
from pathlib import Path
import torch
from torch import nn

from qwen3_config import Qwen3Config

is_prefill = True

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
        
        if config.head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        else:
            head_dim = config.head_dim
        
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=config.rope_base,
            context_length=config.context_length
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.sampler = Sampler()
    
    def forward(self, input_ids, output_ids):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        num_tokens = output_ids.shape[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=hidden_states.device, dtype=torch.bool), diagonal=1)

        global is_prefill
        if is_prefill:
            start_pos = 0
        else:
            start_pos = num_tokens - 1

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, mask, self.cos, self.sin, offset=start_pos)
        hidden_states = self.final_norm(hidden_states)
        if is_prefill:
            x = hidden_states[:, -1:, :].contiguous() # hidden state of last token of each seq
        else:
            x = hidden_states
        logits = self.out_head(x.to(self.config.dtype))
        if is_prefill:
            is_prefill = False
        return logits

    def prepare_sample(self, temperature):
        temperatures = []
        temperatures.append(temperature)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        temperatures = torch.tensor(temperatures, dtype=torch.float32, device=device)
        return temperatures

    def generate(self, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
        output_ids = idx
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = self(idx_cond, output_ids)
            logits = logits[:, -1, :]
               # New: Filter logits with top_k sampling
            if top_k is not None:
                # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
            # New: Apply temperature scaling
            token_ids = None
            if temperature > 0.0:
                temperatures = self.prepare_sample(temperature)
                token_ids = self.sampler(logits, temperatures)
            else:
                token_ids = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            if token_ids == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                break

            output_ids = torch.cat((output_ids, token_ids), dim=1)
            idx = token_ids
    
        return output_ids
    
    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
    ):
        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        if os.path.isdir(repo_id):
            model_path = Path(repo_id)
        else:
            model_path = Path(snapshot_download(repo_id=repo_id))
        tok_file = f"{model_path}/tokenizer.json"

        key = "0.6b"
        if not key in repo_id.lower():
            raise ValueError(f"Could not determine model config from repo_id: {repo_id}")

        config = Qwen3Config()
        model = cls(config)
        
        st_files = list(model_path.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")
        
        weights = {}
        for st_file in st_files:
            with safe_open(st_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)

        from qwen3_weight import load_weights_into_qwen
        load_weights_into_qwen(model, config, weights)
        
        model.to(config.dtype)
        
        return model, tok_file, config

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
            dtype=config.dtype
        )
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
    
    def forward(self, hidden_states, mask, cos, sin, offset=0):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, mask, cos, sin, offset)
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
        dtype=None
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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.attn_cache = KVCache(1, self.num_kv_groups, 512, self.head_dim, device, dtype=dtype)

    def prefill_step(self, hidden_states, mask, cos, sin):
        b, num_tokens, _ = hidden_states.shape
        
        # Apply projections
        query_states = self.q_proj(hidden_states) # [b, num_tokens, num_heads * head_dim]
        key_states = self.k_proj(hidden_states) # [b, num_tokens, num_kv_groups * head_dim]
        value_states = self.v_proj(hidden_states) # [b, num_tokens, num_kv_groups * head_dim]
        
        # Reshape
        query_states = query_states.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        value_states = value_states.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        
        # Optional norm
        if self.q_norm:
            query_states = self.q_norm(query_states)
        if self.k_norm:
            key_states = self.k_norm(key_states)
        # Apply rope
        query_states = apply_rope(query_states, cos, sin)
        key_states = apply_rope(key_states, cos, sin)

        # append K/V to cache
        self.attn_cache.prefill_kv(key_states, value_states)

        # Expand K and V to match number of heads
        key_states = key_states.repeat_interleave(self.group_size, dim=1)
        value_states = value_states.repeat_interleave(self.group_size, dim=1)
        # Attention
        attn_scores = query_states @ key_states.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (attn_weights @ value_states).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        out = self.o_proj(context)
        return out

    def decode_step(self, hidden_states, mask, cos, sin, offset=0):
        b, num_tokens, _ = hidden_states.shape

        assert num_tokens == 1  # only one token forever

        assert is_prefill == False

        # Apply projections
        query_states = self.q_proj(hidden_states) # [b, 1, num_heads * head_dim]
        key_states = self.k_proj(hidden_states) # [b, 1, num_kv_groups * head_dim]
        value_states = self.v_proj(hidden_states) # [b, 1, num_kv_groups * head_dim]
        
        # Reshape
        query_states = query_states.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)  # [b, num_kv_groups, num_tokens, head_dim]
        value_states = value_states.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)  # Xv_BxNxSxH

        # Optional norm
        if self.q_norm:
            query_states = self.q_norm(query_states)
        if self.k_norm:
            key_states = self.k_norm(key_states)

        # Apply rope for query
        query_states = apply_rope(query_states, cos, sin, offset)

        # Apply rope for key
        key_states = apply_rope(key_states, cos, sin, offset)  # Xk_BxNxSxH
        # append K/V to cache
        attn_k, attn_v = self.attn_cache.get_kv_for_attention(key_states, value_states)
        self.attn_cache.update_cache(key_states, value_states)

        # Expand K and V to match number of heads
        K_repeat = attn_k.repeat_interleave(self.group_size, dim=1)  # [batch_size, num_heads, T, head_dim]
        V_repeat = attn_v.repeat_interleave(self.group_size, dim=1)

        # Attention
        attn_scores = query_states @ K_repeat.transpose(2, 3)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (attn_weights @ V_repeat).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        out = self.o_proj(context)
        return out

    def forward(self, hidden_states, mask, cos, sin, offset=0):
        b, num_tokens, _ = hidden_states.shape
        if is_prefill:
            return self.prefill_step(hidden_states, mask, cos, sin)
        else:
            return self.decode_step(hidden_states, mask, cos, sin, offset)

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0) # Shape: [context_length, head_dim // 2]
    # Expand angles to match head dim
    angles = torch.cat([angles, angles], dim=1) # Shape: [context_length, head_dim]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    return cos, sin

def apply_rope(x, cos, sin, offset=0):
    # x: [batch_size, num_heads, seq_len, head_dim]
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2:]
    
    cos = cos[offset:offset+seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim // 2)
    sin = sin[offset:offset+seq_len, :].unsqueeze(0).unsqueeze(0)
    
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    
    return x_rotated.to(dtype=x.dtype)

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
