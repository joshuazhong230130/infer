import torch
from qwen3_config import Qwen3Config

def load_weights_into_qwen(model, config:Qwen3Config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor: '{tensor_name}'. Left: {left.shape}, right: {right.shape}")
        
        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right)
            else:
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))
        
        return left
    
    model.embed_tokens.weight = assign(model.embed_tokens.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
    
    for l in range(config.num_hidden_layers):
        layer = model.layers[l]
        att = layer.self_attn
        
        att.q_proj.weight = assign(
            att.q_proj.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        att.k_proj.weight = assign(
            att.k_proj.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        att.v_proj.weight = assign(
            att.v_proj.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )
        att.o_proj.weight = assign(
            att.o_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )
        # QK norms
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.scale = assign(
                att.q_norm.scale,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight"
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.scale = assign(
                att.k_norm.scale,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight"
            )
        # Attention layernorm
        layer.input_layernorm.scale = assign(
            layer.input_layernorm.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )
        # FeedForward 
        layer.mlp.gate_proj.weight = assign(
            layer.mlp.gate_proj.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        layer.mlp.up_proj.weight = assign(
            layer.mlp.up_proj.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        layer.mlp.down_proj.weight = assign(
            layer.mlp.down_proj.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )

        layer.post_attention_layernorm.scale = assign(
            layer.post_attention_layernorm.scale,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    model.final_norm.scale = assign(
        model.final_norm.scale, 
        params["model.norm.weight"],
        "model.norm.weight"
    )
    if "lm_head.weight" in params:
        model.out_head.weight = assign(
            model.out_head.weight,
            params["lm_head.weight"],
            "lm_head.weight"
        )
    else:
        model.out_head.weight = model.embed_tokens.weight
        print("Model uses weight tying")
