import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


@torch.no_grad()
def get_word_probabilities(sentence: str, word: str) -> float:
    # 1. Prepare input
    prompt = sentence.strip() + " "  # Ensure there's a space before the sentence
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]  # Shape: [1, sequence_length]

    # 1. Tokenize the target word with a leading space
    target_word = " " + word.strip()
    target_ids = tokenizer.encode(target_word, add_special_tokens=False) # List of token IDs for the target word (provided with the context)

    if not target_ids:
        return None  # Handle case where the target word cannot be tokenized

    # 3. Apply the model to the model adn apply the softmax prob to the prob distribution
    # transform the input with the model into with, for memory efficiency
    total_logp = 0.0
    with torch.no_grad():
        for token_id in target_ids:
            # We pass the full input_ids each time
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0, -1, :] 
            
            # Use log_softmax for better stability
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Add to our running total
            total_logp += log_probs[token_id].item()
            
            # Update input_ids for the next token in the phrase
            new_id = torch.tensor([[token_id]], device=model.device)
            input_ids = torch.cat([input_ids, new_id], dim=1)
        

    # Return the total log probability of the target word given the sentence
    return total_logp  
