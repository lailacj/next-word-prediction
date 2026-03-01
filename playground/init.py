import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="google-bert/bert-base-uncased",
    dtype=torch.float16,
    device=0
)
output = pipeline("Plants create [MASK] through a process known as photosynthesis.")
