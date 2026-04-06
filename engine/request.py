
from copy import copy
import enum

from sample_param import SampleParam

class RequestStatus(enum.IntEnum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED = enum.auto()

class Request:
    def __init__(self, request_id: str, prompt: str, token_ids: list[int], sample_param:SampleParam):
        self.request_id = request_id
        self.prompt = prompt
        self.output = []
        self.token_ids = copy(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_tokens = len(token_ids)
        self.last_token_id = token_ids[-1]
        self.status = RequestStatus.WAITING
        self.temperature = sample_param.temperature
        self.top_k = sample_param.top_k
        self.max_output_tokens = sample_param.max_output_tokens

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token_id = token_id
        self.num_tokens += 1

    @property
    def is_finished(self):
        return self.status == RequestStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]
