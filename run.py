import time
import torch
import argparse

from qwen3 import Qwen3Model 
from tokenizer import Qwen3Tokenizer

def main(prompt, model_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tok_file, config = Qwen3Model.from_pretrained(model_id)
    model.to(device)

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tok_file,
        add_generation_prompt=True,
        add_thinking=False
    )

    input_token_ids = tokenizer.encode(prompt)

    torch.manual_seed(123)

    start = time.time()
    
    output_token_ids = model.generate(
        idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
        max_new_tokens=100,
        context_size=config.context_length,
        top_k=30,
        temperature=0.6,
        eos_id=tokenizer.eos_token_id
    )
    
    total_time = time.time() - start
    print(f"Time: {total_time:.2f} sec")
    print(f"{int(len(output_token_ids[0])/total_time)} tokens/sec")
    
    
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
