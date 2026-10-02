import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# initilialize the deepseek model and the tokenizer
#model_name = "deepseek-ai/DeepSeek-R1"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)


@torch.no_grad()
def tokenize_sentence(sentence: str):
    """
    Tokenize the sentence with the DeepSeek chat template.
    """
    messages = [{"role": "user", "content": sentence.strip()}]
    inputs = tokenizer(sentence.strip(), return_tensors="pt").to(model.device)
    return inputs


@torch.no_grad()
def tokenize_word(word: str):
    """
    Tokenize the target word without special tokens.
    """
    #add the space prefix to symblolize that this is a continuation of the sentence (next word)
    target_word = " " + word.strip()
    target_ids = tokenizer.encode(target_word, add_special_tokens=False)

    if not target_ids:
        return None

    return target_ids


@torch.no_grad()
def get_word_probabilities(sentence_token, word_token_ids) -> float:
    """
    Compute the total log probability of a word given the DeepSeek prompt.
    """
    total_logp = 0.0

    for token_id in word_token_ids:
        # keep updating the sentence token ids by appending the previously predicted token, so that we can get the correct probability for multi-token words
        outputs = model(**sentence_token)
        logits = outputs.logits[0, -1, :]

        # calculate log probabilities and extract the log probability for the current token_id
        log_probs = F.log_softmax(logits, dim=-1)
        total_logp += log_probs[token_id].item()
      
    return total_logp