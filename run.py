import time
import torch

from sample_param import SampleParam
from tokenizer import Qwen3Tokenizer
from engine.engine_core import EngineCore

def main(prompt, model_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    engine_core = EngineCore(model_id, device)

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=engine_core.tokenizer_file,
        add_generation_prompt=True,
        add_thinking=False
    )

    sample_param = SampleParam(temperature=0.6, top_k=30, max_output_tokens=200)
    torch.manual_seed(123)

    start = time.time()
    
    output_token_ids = engine_core.generate(
        prompt=prompt,
        sample_param=sample_param,
        eos_id=tokenizer.eos_token_id
    )
    
    total_time = time.time() - start
    # print(f"Time: {total_time:.2f} sec")
    # print(f"{int(len(output_token_ids[0])/total_time)} tokens/sec")
    
    
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")
    
    output_text = tokenizer.decode(output_token_ids)
    
    print("\n\nOutput text:\n\n", output_text + "...")

if __name__ == "__main__":
    prompt="Can pigs fly? You should explain the details."
    model_id=r"D:\inteligens\models\Qwen3-0.6B"
    main(prompt, model_id)
