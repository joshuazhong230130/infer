import torch

class Qwen3Config:
    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
    ):
        QWEN_CONFIG_06_B = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 1024,                # hidden_size
            "n_heads": 16,                  # Attention heads num_attention_heads
            "n_layers": 28,                 # num_hidden_layers
            "hidden_dim": 3072,             # Intermediate dimension in FeedForward intermediate_size
            "head_dim": 128,                # Size of the heads in GQA  head_dim
            "qk_norm": True,                # Whether to normalize queries and keys in GQA
            "n_kv_groups": 8,               # Key-value groups for GQA  num_key_value_heads
            "rope_base": 1_000_000.0,       # The base in ropes "theta"
            "dtype": torch.bfloat16,
        }

        self.vocab_size = QWEN_CONFIG_06_B["vocab_size"]
        self.hidden_size = QWEN_CONFIG_06_B["emb_dim"]
        self.intermediate_size = QWEN_CONFIG_06_B["hidden_dim"]
        self.num_hidden_layers = QWEN_CONFIG_06_B["n_layers"]
        self.num_attention_heads = QWEN_CONFIG_06_B["n_heads"]
        self.num_key_value_heads = QWEN_CONFIG_06_B["n_kv_groups"]
        self.head_dim = QWEN_CONFIG_06_B["head_dim"]
        self.context_length = QWEN_CONFIG_06_B["context_length"]
        self.rope_base = QWEN_CONFIG_06_B["rope_base"]
        self.dtype = QWEN_CONFIG_06_B["dtype"]
        self.qk_norm = True
