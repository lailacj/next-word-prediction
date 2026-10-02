import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


@torch.no_grad()
def tokenize_sentence(sentence: str) :
    """
    Prepare the input sentence, tokenize it, and return the token IDs of the tensor space. 
    We add a space before the sentence to ensure proper tokenization of the last word.
    The
    """
    prompt = sentence.strip() + " "  # Ensure there's a space before the sentence
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]  # Shape: [1, sequence_length]

    return input_ids

@torch.no_grad()
def tokenize_word(word:str): 
    """
    Tokenize the word into a tensor space of token ids
    """
    target_word = " " + word.strip()
    target_ids = tokenizer.encode(target_word, add_special_tokens=False) # List of token IDs for the target word (provided with the context)

    if not target_ids:
        return None  # Handle case where the target word cannot be tokenized

    return target_ids

@torch.no_grad()
def get_word_probabilities(sentenc_token_ids, word_token_ids) -> float:
    """
    Apply the model to the model and apply the softmax prob to the prob distribution
    then transform the input with the model into with, for memory efficiency
    """
    total_logp = 0.0
    
    for token_id in word_token_ids:
        # We pass the full input_ids each time
        outputs = model(input_ids=sentenc_token_ids)
        logits = outputs.logits[0, -1, :] 
        
        # Use log_softmax for better stability
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Add to our running total
        total_logp += log_probs[token_id].item()
            
        # Update input_ids for the next token in the phrase
        new_id = torch.tensor([[token_id]], device=model.device)
        sentenc_token_ids = torch.cat([sentenc_token_ids, new_id], dim=1)
        

    # Return the total log probability of the target word given the sentence
    return total_logp  
