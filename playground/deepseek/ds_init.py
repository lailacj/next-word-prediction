# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# inputs = tokenizer.apply_chat_template(
# 	messages,
# 	add_generation_prompt=True,
# 	tokenize=True,
# 	return_dict=True,
# 	return_tensors="pt",
# ).to(model.device)

# outputs = model.generate(**inputs, max_new_tokens=40)
# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# 1. Define model and load tokenizer/model
# We use a distilled version (1.5B) for efficiency, but you can use any R1 variant.
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
#model_name = "deepseek-ai/DeepSeek-R1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def get_next_word_probability(sentence, target_word):
    # 2. Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)

    # 3. Tokenize the target word to find its ID
    # Note: We add a space prefix because models often treat " word" differently than "word"
    target_tokens = tokenizer.encode(" " + target_word.strip(), add_special_tokens=False)
    print(target_tokens)
    target_token_id = target_tokens[0] # Taking the first token of the target word

    # 4. Perform forward pass to get logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits # Shape: [batch, sequence_length, vocab_size]
    
    # 5. Get the distribution for the NEXT token (after the last input token)
    last_token_logits = logits[0, -1, :] 
    probabilities = F.log_softmax(last_token_logits, dim=-1)
    
    # 6. Extract the specific probability for your target word
    target_prob = probabilities[target_token_id].item()
    
    return target_prob, target_token_id

# Example Usage
sentence = "The cat is on the"
target = "roofman"
prob, token_id = get_next_word_probability(sentence, target)

print(f"Sentence: '{sentence}'")
print(f"Target word: '{target}' (Token ID: {token_id})")
print(f"Probability: {prob:.4f} ({prob*100:.2f}%)")