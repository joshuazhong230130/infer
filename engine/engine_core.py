import torch
import os
from pathlib import Path
from qwen3 import Qwen3Model
from qwen3_config import Qwen3Config
from layers.sampler import Sampler
from qwen3_weight import load_weights_into_qwen
from huggingface_hub import snapshot_download
from safetensors import safe_open


class EngineCore:
    def __init__(self, model_id: str):
        self.config = None
        self.sampler = Sampler()
        self.load_model(model_id)

    def prepare_sample(self, temperature):
        temperatures = []
        temperatures.append(temperature)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        temperatures = torch.tensor(temperatures, dtype=torch.float32, device=device)
        return temperatures

    def generate(self, idx, max_new_tokens, temperature=0.0, top_k=None, eos_id=None):
        output_ids = idx
        context_size = self.config.context_length
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = self.model(idx_cond, output_ids)
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

    def load_model(
        self,
        repo_id: str,
    ):
        if os.path.isdir(repo_id):
            model_path = Path(repo_id)
        else:
            model_path = Path(snapshot_download(repo_id=repo_id))
        self.tok_file = f"{model_path}/tokenizer.json"

        key = "0.6b"
        if not key in repo_id.lower():
            raise ValueError(f"Could not determine model config from repo_id: {repo_id}")

        self.config = Qwen3Config()
        self.model = Qwen3Model(self.config)


        st_files = list(model_path.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")
        
        weights = {}
        for st_file in st_files:
            with safe_open(st_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)

        load_weights_into_qwen(self.model, self.config, weights)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(dtype=self.config.dtype, device=device)
