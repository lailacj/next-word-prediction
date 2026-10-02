import torch
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os
from dotenv import load_dotenv

# Load variables from .env into the environment
load_dotenv()

LLAMA_TOKEN = os.getenv("LLAMA_TOKEN")

# Paste your token here
login(token=LLAMA_TOKEN)

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


@torch.no_grad()
def tokenize_sentence(sentence: str):
    """
    Tokenize the sentence prefix used for next-word prediction.
    """
    prompt = sentence.strip()
    inputs = tokenizer(prompt, return_tensors="pt")

    return inputs["input_ids"]


@torch.no_grad()
def tokenize_word(word: str):
    """
    Tokenize the target word without special tokens.
    """
    target_word = " " + word.strip()
    target_ids = tokenizer.encode(target_word, add_special_tokens=False)

    if not target_ids:
        return None

    return target_ids

@torch.no_grad()
def get_word_probabilities(sentence_token_ids, word_token_ids) -> float:
    total_log_prob = 0.0
    input_ids = sentence_token_ids
    past_key_values = None # This stores the "memory" of the sentence

    for token_id in word_token_ids:
        # We only pass the NEW token if we have a cache (past_key_values)
        # For the first iteration, we pass the whole sentence
        outputs = model(
            input_ids=input_ids, 
            past_key_values=past_key_values, 
            use_cache=True
        )
        
        logits = outputs.logits[0, -1, :]
        past_key_values = outputs.past_key_values # Update the cache

        # Log_softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        total_log_prob += log_probs[token_id].item()

        # Update input_ids to ONLY the next token for the next loop
        input_ids = torch.tensor([[token_id]], device=input_ids.device)

    return total_log_prob