import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchview import draw_graph
import math

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def get_word_probabilities(sentence: str, word: str) -> float:
    # 1. Prepare input
    prompt = sentence.strip() + " "  # Ensure there's a space before the sentence
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(inputs.keys())
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
            logits = outputs.logits[0, -1, :] # Last token distribution
            
            # Use log_softmax for better stability
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Add to our running total
            total_logp += log_probs[token_id].item()
            
            # Update input_ids for the next token in the phrase
            new_id = torch.tensor([[token_id]], device=model.device)
            input_ids = torch.cat([input_ids, new_id], dim=1)
        
    # before you leave create a graph
    visualize_the_graph(inputs)
    return {
        "prob": math.exp(total_logp),  # Convert log probability back to regular probability
        "log_prob": total_logp,
        "sentence": sentence,
        "word": word,
    }


def visualize_the_graph(inputs):
    # This will show the high-level hierarchy of Qwen's 28 layers
    model_graph = draw_graph(model, input_data=inputs["input_ids"], expand_nested=False)
    model_graph.visual_graph.render("qwen_hierarchy", format="png")

