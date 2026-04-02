import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "deepseek-ai/DeepSeek-R1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.eval()


@torch.no_grad()
def tokenize_sentence(sentence: str):
    """
    Tokenize the sentence with the DeepSeek chat template.
    """
    messages = [{"role": "user", "content": sentence.strip()}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

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
    """
    Compute the total log probability of a word given the DeepSeek prompt.
    """
    total_logp = 0.0

    for token_id in word_token_ids:
        outputs = model(input_ids=sentence_token_ids)
        logits = outputs.logits[0, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        total_logp += log_probs[token_id].item()

        new_id = torch.tensor([[token_id]], device=model.device)
        sentence_token_ids = torch.cat([sentence_token_ids, new_id], dim=1)

    return total_logp