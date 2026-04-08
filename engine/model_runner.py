
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors import safe_open
import torch
from engine.request import Request
from layers.sampler import Sampler
from qwen3 import Qwen3Model
from qwen3_config import Qwen3Config
from qwen3_weight import load_weights_into_qwen

class ModelRunner:
    def __init__(self, model_id: str, device):
        self.config = None
        self.model_id = model_id
        self.model = None
        self.tok_file = None
        self.device = device
        self.sampler = Sampler()
        self.load_model(model_id)

    def prepare_prefill(self, requests: list[Request]):
        token_ids = []
        positions = []
        for request in requests:
            token_ids.extend(request.token_ids)
            seq_len = len(request.token_ids)
            positions.extend(list(range(seq_len)))
        return token_ids, positions

    def prepare_decode(self, requests: list[Request]):
        token_ids = []
        positions = []
        for request in requests:
            token_ids.append(request.last_token_id)
            seq_len = len(request.token_ids)
            positions.append(seq_len - 1)
        return token_ids, positions

    def prepare_sample(self, temperature):
        temperatures = []
        temperatures.append(temperature)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        temperatures = torch.tensor(temperatures, dtype=torch.float32, device=device)
        return temperatures

    def sampling(self, logits, temperature, top_k=None):
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

        return token_ids

    def run(self, requests: list[Request], is_prefill: bool) -> list[int]:
        token_ids_batch = []

        if is_prefill:
            token_ids, positions = self.prepare_prefill(requests)
        else:
            token_ids, positions = self.prepare_decode(requests)
        input_token_ids=torch.tensor(token_ids, device=self.device).unsqueeze(0)

        logits = self.model(input_token_ids, positions, is_prefill=is_prefill)
        logits = logits[:, -1, :]
        request = requests[0]
        token_id = self.sampling(logits, request.temperature, request.top_k)
        token_ids_batch.append(token_id)

        return token_ids_batch

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
        self.model.to(dtype=self.config.dtype, device=self.device)


    @property
    def tokenizer_file(self):
        return self.tok_file
