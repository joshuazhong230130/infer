
class EngineCoreRequest:
    request_id: str
    prompt_token_ids: list[int]

class EngineCoreOutput:
    request_id: str
    new_token_ids: list[int]
