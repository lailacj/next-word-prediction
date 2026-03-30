import torch
from transformers import pipeline

# Bert pipeline
pipeline = pipeline(
    task="fill-mask",
    model="google-bert/bert-base-uncased",
    dtype=torch.float16,
    device=0
)
output = pipeline("Plants create [MASK] through a process known as photosynthesis.")


# llama pipeline
pipe_llama = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")


# deepseek pipeline# Use a pipeline as a high-level helper
pipe_deepseek = pipeline(
    "text-generation", 
    model="deepseek-ai/DeepSeek-R1", 
    trust_remote_code=True
)

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe_deepseek(messages)
