from collections import deque
from engine.request import Request, RequestStatus

class Scheduler:
    def __init__(self):
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()
        self.eos_id:int = None

    def add_request(self, request: Request):
        self.waiting.append(request)

    @property
    def eos_token_id(self):
        return self.eos_id

    @eos_token_id.setter
    def eos_token_id(self, value: int):
        self.eos_id = value

    def schedule(self)->tuple[list[Request], bool]:
        scheduled_requests = []
        while self.waiting:
            request = self.waiting.popleft()
            request.status = RequestStatus.RUNNING
            scheduled_requests.append(request)
            self.running.append(request)

        if scheduled_requests:
            return scheduled_requests, True

        while self.running:
            request = self.running.popleft()
            scheduled_requests.append(request)
        
        self.running.extendleft(scheduled_requests)

        return scheduled_requests, False

    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0

    def postprocess(self, requests: list[Request], token_ids: list[int]):
        for request, token_id in zip(requests, token_ids):
            request.append_token(token_id)
            if (token_id == self.eos_id) or request.num_completion_tokens == request.max_output_tokens:
                request.status = RequestStatus.FINISHED
                self.running.remove(request)

