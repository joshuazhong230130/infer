import torch
from torch import nn
import torch.nn.functional as F

def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float = None,
) -> torch.Tensor:
    total_tokens, n_heads, head_dim = q.shape
    batch_size = len(cu_seqlens) - 1

    if scale is None:
        scale = head_dim ** (-0.5)

    # 1. 构造全局下三角 causal mask（自回归因果注意力）
    # 先生成全局位置 id：每个token属于第几个序列、序列内局部下标
    seq_idx = torch.empty(total_tokens, dtype=torch.int32, device=q.device)
    for b in range(batch_size):
        start = cu_seqlens[b]
        end = cu_seqlens[b + 1]
        seq_idx[start:end] = torch.arange(end - start, dtype=torch.int32, device=q.device)

    # 构造 [max_seqlen, max_seqlen] 因果下三角 mask
    mask = torch.ones((max_seqlen, max_seqlen), device=q.device, dtype=torch.bool)
    mask = torch.tril(mask)  # 只允许看到自身及之前token
    mask = mask.logical_not()  # True = 需要mask掉

    # 2. Q @ K^T 注意力分数
    # 构造大矩阵：[n_heads, max_seqlen, max_seqlen]
    attn_score = torch.zeros(
        (n_heads, max_seqlen, max_seqlen),
        device=q.device,
        dtype=q.dtype
    )

    # 按每个序列单独填充分数，互不干扰（变长核心）
    for b in range(batch_size):
        start = cu_seqlens[b]
        end = cu_seqlens[b + 1]
        length = end - start
        # 当前序列 q/k/v
        q_b = q[start:end].transpose(0, 1)  # [n_heads, length, hd]
        k_b = k[start:end].transpose(0, 1)  # [n_heads, length, hd]

        # 单序列注意力分数
        score_b = torch.matmul(q_b, k_b.transpose(-2, -1)) * scale
        attn_score[:, :length, :length] = score_b

    # 3. 施加因果mask
    attn_score.masked_fill_(mask, -torch.inf)

    # 4. softmax
    attn_weight = F.softmax(attn_score, dim=-1)

    # 5. 权重乘V
    out = torch.zeros_like(q)
    for b in range(batch_size):
        s = cu_seqlens[b]
        e = cu_seqlens[b + 1]
        L = e - s

        v_b = v[s:e].transpose(0, 1)  # [n_heads, L, hd]
        out_b = torch.matmul(attn_weight[:, :L, :L], v_b)
        out_b = out_b.transpose(0, 1)  # [L, n_heads, hd]
        out[s:e] = out_b

    return out

def attention_with_kvcache(
    q: torch.Tensor,                # [batch_size, n_heads, head_dim]
    k_cache: torch.Tensor,          # [batch_size, max_kv_tokens, n_heads, head_dim]
    v_cache: torch.Tensor,          # [batch_size, max_kv_tokens, n_heads, head_dim]
    cache_seq_lens: torch.Tensor,   # [batch_size]
    scale: float = None,
) -> torch.Tensor:
    B, H, D = q.shape
    out = torch.zeros_like(q)  # [B, H, D]

    if scale is None:
        scale = D ** (-0.5)

    for b in range(B):
        L = cache_seq_lens[b]
        if L == 0:
            continue
        # [L, H, D] → [H, L, D]
        k = k_cache[b, :L].permute(1, 0, 2) # [H, L, D]
        v = v_cache[b, :L].permute(1, 0, 2) # [H, L, D]
        q_b = q[b].unsqueeze(1) # [H, 1, D]
        attn_score = torch.matmul(q_b, k.transpose(-1, -2)).squeeze(1) # [H, L]
        attn_score = attn_score * scale
        attn_weight = F.softmax(attn_score, dim=-1)
        # [H, 1, L] @ [H, L, D] → [H, D]
        out[b] = torch.matmul(attn_weight.unsqueeze(1), v).squeeze(1) # [H, D]

    return out

class Attention(nn.Module):
    def __init__(self,
        d_in,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.scale = scale
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads

    def forward(self, query : torch.Tensor, key : torch.Tensor, value : torch.Tensor, is_prefill=True):
        pass
