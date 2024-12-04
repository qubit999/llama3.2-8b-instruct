import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()

login(token=os.getenv("HUGGINGFACE_TOKEN"))

model_name = "meta-llama/Llama-3.2-3B-Instruct"
cache = "./cache"

# Load tokenizer and model with optimizations
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache,
    torch_dtype=torch.float16,  # Load in half precision
    low_cpu_mem_usage=True,     # Reduce memory usage during loading
)

# Save locally
model.save_pretrained("./Llama-3.2-3B-Instruct")
tokenizer.save_pretrained("./Llama-3.2-3B-Instruct")