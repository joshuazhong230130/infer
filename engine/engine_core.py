import uuid
from engine.model_runner import ModelRunner

from sample_param import SampleParam
from tokenizer import Qwen3Tokenizer
from engine.scheduler import Scheduler
from engine.request import Request


class EngineCore:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.device = device
        self.scheduler = Scheduler()
        self.model_runner = ModelRunner(model_id, self.device)
        self.tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=self.model_runner.tokenizer_file,
            add_generation_prompt=True,
            add_thinking=False
        )

    def add_request(self, prompt: str, sample_param:SampleParam):
        token_ids = self.tokenizer.encode(prompt)
        request = Request(request_id=str(uuid.uuid4()), prompt=prompt, token_ids=token_ids, sample_param=sample_param)
        self.scheduler.add_request(request)
        return request

    def step(self):
        requests, is_prefill = self.scheduler.schedule()
        token_ids_batch = self.model_runner.run(requests, is_prefill)
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

    @property
    def tokenizer_file(self):
        return self.model_runner.tokenizer_file
