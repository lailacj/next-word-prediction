from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM
from huggingface_hub import login
import os
from dotenv import load_dotenv

# ABSTRACT INTERFACE FOR LANGUAGE MODELS
class LanguageModel(ABC):
    @abstractmethod
    def tokenize_sentense(self, sentence: str):
        pass

    @abstractmethod
    def tokenize_word(self, word: str):
        pass

    @property
    @abstractmethod
    def priority(self):
        pass

    @abstractmethod
    def predict_next_word(self, sentence_token_ids, word_token_ids):
        pass

    @abstractmethod
    def get_ouptut_file(self):
        pass


# QWEN MODEL IMPLEMENTATION
class QwenModel(LanguageModel):
    def __init__(self):
        # initilialize the qwen model and the tokenizer
        self.model_name = "Qwen/Qwen2.5-7B"
        self.output_file = "../data/qwen/qwen_data.csv"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # write up the header of the outpul as you initialized the model
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("sentence_num,sentence,word,qwen_prob\n")

    @property
    def priority(self):
        return 1

    @torch.no_grad()
    def tokenize_sentense(self, sentence: str):
        """
        Prepare the input sentence, tokenize it, and return the token IDs of the tensor space.
        We add a space before the sentence to ensure proper tokenization of the last word.
        The
        """
        prompt = sentence.strip() + " "  # Ensure there's a space before the sentence
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]  # Shape: [1, sequence_length]

        return input_ids

    @torch.no_grad()
    def tokenize_word(self, word: str):
        """
        Tokenize the word into a tensor space of token ids
        """
        target_word = " " + word.strip()
        target_ids = self.tokenizer.encode(target_word, add_special_tokens=False)  # List of token IDs for the target word (provided with the context)

        if not target_ids:
            return None  # Handle case where the target word cannot be tokenized

        return target_ids

    @torch.no_grad()
    def predict_next_word(self, sentenc_token_ids, word_token_ids) -> float:
        """
        Apply the model to the model and apply the softmax prob to the prob distribution
        then transform the input with the model into with, for memory efficiency
        """
        total_logp = 0.0

        for token_id in word_token_ids:
            # We pass the full input_ids each time
            outputs = self.model(input_ids=sentenc_token_ids)
            logits = outputs.logits[0, -1, :]

            # Use log_softmax for better stability
            log_probs = F.log_softmax(logits, dim=-1)

            # Add to our running total
            total_logp += log_probs[token_id].item()

            # Update input_ids for the next token in the phrase
            new_id = torch.tensor([[token_id]], device=self.model.device)
            sentenc_token_ids = torch.cat([sentenc_token_ids, new_id], dim=1)

        # Return the total log probability of the target word given the sentence
        return total_logp

    def get_ouptut_file(self):
        return self.output_file


# BERT MODEL IMPLEMENTATION
class BertModel(LanguageModel):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking")
        self.model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-large-uncased-whole-word-masking")
        self.output_file = "../data/bert_data/bert_cloze_output.csv"

    @property
    def priority(self):
        return 2

    def tokenize_sentense(self, sentence: str):
        return self.tokenizer(sentence + " [MASK].", return_tensors="pt")

    def tokenize_word(self, word: str):
        target_word = " " + word.strip()
        target_ids = self.tokenizer.encode(target_word, add_special_tokens=False)
        if not target_ids:
            return None
        return target_ids

    def get_next_word_probability_distribution(self, sentence):
        '''return the probs of all next tokens/words wiht their associated probs gdiven"
        the input sentence with a [MASK] token at the end. '''
        # Returns all next tokens and their probabilities.
        inputs = self.tokenizer(sentence + " [MASK].", return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits

        mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        mask_token_logits = logits[0, mask_token_index, :].squeeze()
        probs = torch.softmax(mask_token_logits, dim=-1)
        predicted_tokens = self.tokenizer.convert_ids_to_tokens(range(len(mask_token_logits)))

        # Return the results as a list of (token, probability) tuples
        return list(zip(predicted_tokens, probs.tolist()))

    def get_specific_word_probability(self, sentence_num, sentence, word_list, next_word_probabilities):
        '''This function searches the BERT probability distribution for specific words and
        stores the results in a dictionary.'''
        for token, prob in next_word_probabilities:
            if token not in word_list:
                continue
            else:
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(f"{sentence_num},{sentence},{token},{prob}\n")

    def predict_next_word(self, sentence_token_ids, word_token_ids):
        return self.get_next_word_probability_distribution(sentence_token_ids)

    


# DEEPSEEK MODEL IMPLEMENTATION
class DeepSeekModel(LanguageModel):
    def __init__(self):
        # initilialize the deepseek model and the tokenizer
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        self.output_file = "../deepseek/deepseek_data.csv"

        # write up the header of the outpul as you initialized the model
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("sentence_num,sentence,word,deepseek_prob\n")
       
        

    @property
    def priority(self):
        return 3

    @torch.no_grad()
    def tokenize_sentense(self, sentence: str):
        """
        Tokenize the sentence with the DeepSeek chat template.
        """
        messages = [{"role": "user", "content": sentence.strip()}]
        inputs = self.tokenizer(sentence.strip(), return_tensors="pt").to(self.model.device)
        return inputs

    @torch.no_grad()
    def tokenize_word(self, word: str):
        """
        Tokenize the target word without special tokens.
        """
        #add the space prefix to symblolize that this is a continuation of the sentence (next word)
        target_word = " " + word.strip()
        target_ids = self.tokenizer.encode(target_word, add_special_tokens=False)

        if not target_ids:
            return None

        return target_ids

    @torch.no_grad()
    def predict_next_word(self, sentence_token, word_token_ids) -> float:
        """
        Compute the total log probability of a word given the DeepSeek prompt.
        """
        total_logp = 0.0

        for token_id in word_token_ids:
            # keep updating the sentence token ids by appending the previously predicted token, so that we can get the correct probability for multi-token words
            outputs = self.model(**sentence_token)
            logits = outputs.logits[0, -1, :]

            # calculate log probabilities and extract the log probability for the current token_id
            log_probs = F.log_softmax(logits, dim=-1)
            total_logp += log_probs[token_id].item()

        return total_logp

    def get_ouptut_file(self):
        return self.output_file


# LLAMA MODEL IMPLEMENTATION
class LlamaModel(LanguageModel):
    def __init__(self):
        # Load variables from .env into the environment
        load_dotenv()

        # connect/login to the llam model using huggingface
        LLAMA_TOKEN = os.getenv("LLAMA_TOKEN")
        # Paste your token here
        login(token=LLAMA_TOKEN)

        # initialize the model and its tokenize
        self.model_name = "meta-llama/Llama-3.2-1B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.output_file = "../data/llama/llama_data.csv"

        # write up the header of the outpul as you initialized the model
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("sentence_num,sentence,word,llama_prob\n")

    @property
    def priority(self):
        return 4

    @torch.no_grad()
    def tokenize_sentense(self, sentence: str):
        """
        Tokenize the sentence prefix used for next-word prediction.
        """
        prompt = sentence.strip()
        inputs = self.tokenizer(prompt, return_tensors="pt")

        return inputs["input_ids"]

    @torch.no_grad()
    def tokenize_word(self, word: str):
        """
        Tokenize the target word without special tokens.
        """
        target_word = " " + word.strip()
        target_ids = self.tokenizer.encode(target_word, add_special_tokens=False)

        if not target_ids:
            return None

        return target_ids

    @torch.no_grad()
    def predict_next_word(self, sentence_token_ids, word_token_ids) -> float:
        total_log_prob = 0.0
        input_ids = sentence_token_ids
        past_key_values = None  # This stores the "memory" of the sentence

        for token_id in word_token_ids:
            # We only pass the NEW token if we have a cache (past_key_values)
            # For the first iteration, we pass the whole sentence
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits[0, -1, :]
            past_key_values = outputs.past_key_values  # Update the cache

            # Log_softmax for numerical stability
            log_probs = F.log_softmax(logits, dim=-1)
            total_log_prob += log_probs[token_id].item()

            # Update input_ids to ONLY the next token for the next loop
            input_ids = torch.tensor([[token_id]], device=input_ids.device)

        return total_log_prob

    def get_ouptut_file(self):
        return self.output_file
