import time
import torch

from tokenizer import Qwen3Tokenizer
from engine.engine_core import EngineCore

def main(prompt, model_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    engine_core = EngineCore(model_id)

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=engine_core.tok_file,
        add_generation_prompt=True,
        add_thinking=False
    )

    input_token_ids = tokenizer.encode(prompt)

    torch.manual_seed(123)

    start = time.time()
    
    output_token_ids = engine_core.generate(
        idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
        max_new_tokens=200,
        top_k=30,
        temperature=0.6,
        eos_id=tokenizer.eos_token_id
    )
    
    total_time = time.time() - start
    # print(f"Time: {total_time:.2f} sec")
    # print(f"{int(len(output_token_ids[0])/total_time)} tokens/sec")
    
    
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")
    
    output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())
    
    print("\n\nOutput text:\n\n", output_text + "...")

if __name__ == "__main__":
    prompt="Can pigs fly? You should explain the details."
    model_id=r"D:\inteligens\models\Qwen3-0.6B"
    main(prompt, model_id)
