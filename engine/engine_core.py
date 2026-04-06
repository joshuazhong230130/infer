import uuid
import torch
import os
from pathlib import Path
from qwen3 import Qwen3Model
from qwen3_config import Qwen3Config
from layers.sampler import Sampler
from qwen3_weight import load_weights_into_qwen
from huggingface_hub import snapshot_download
from safetensors import safe_open

from sample_param import SampleParam
from tokenizer import Qwen3Tokenizer
from engine.scheduler import Scheduler
from engine.request import Request


class EngineCore:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.config = None
        self.device = device
        self.sampler = Sampler()
        self.load_model(model_id)
        self.tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=self.tok_file,
            add_generation_prompt=True,
            add_thinking=False
        )
        self.scheduler = Scheduler()

    def prepare_sample(self, temperature):
        temperatures = []
        temperatures.append(temperature)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        temperatures = torch.tensor(temperatures, dtype=torch.float32, device=device)
        return temperatures

    def add_request(self, prompt: str, sample_param:SampleParam):
        token_ids = self.tokenizer.encode(prompt)
        request = Request(request_id=str(uuid.uuid4()), prompt=prompt, token_ids=token_ids, sample_param=sample_param)
        self.scheduler.add_request(request)
        return request

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

    def step(self):
        requests, is_prefill = self.scheduler.schedule()
        token_ids_batch = []
        for request in requests:
            if is_prefill:
                idx = request.token_ids
            else:
                idx = request.token_ids[-1:]
            idx=torch.tensor(idx, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(idx, len(request.token_ids))
            logits = logits[:, -1, :]
            token_id = self.sampling(logits, request.temperature, request.top_k)
            token_ids_batch.append(token_id)

            # if token_id != request.eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            #     request.token_ids += token_id.squeeze(0).tolist()

        self.scheduler.postprocess(requests, token_ids_batch)
        output_ids_batch = [request.last_token_id for request in requests if request.is_finished]

        return output_ids_batch

    def generate(self, prompt, sample_param:SampleParam, eos_id: int):
        self.scheduler.eos_token_id = eos_id
        request = self.add_request(prompt, sample_param)
        while not self.scheduler.is_finished():
            self.step()
        output_ids = request.token_ids
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
        self.model.to(dtype=self.config.dtype, device=self.device)
