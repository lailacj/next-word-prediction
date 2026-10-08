"""One interface for the model families supported by the scoring pipeline."""

import os

MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-7B",
    "bert": "google-bert/bert-large-uncased-whole-word-masking",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "llama": "meta-llama/Llama-3.2-1B",
}


class LanguageModel:
    """Load a supported model and score candidate continuations in log space.

    BERT scores one masked token. Causal models condition each continuation
    token on the sentence and the preceding continuation tokens.
    """

    def __init__(self, name: str, *, device="cpu", dtype="float32", revision="main"):
        if name not in MODEL_IDS:
            raise ValueError(f"Unknown model {name!r}. Choose from: {', '.join(MODEL_IDS)}")
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"Unsupported dtype: {dtype}")

        import torch
        from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

        requested_device = torch.device(device)
        if requested_device.type not in ("cpu", "cuda", "mps"):
            raise ValueError("Device must be cpu, cuda, cuda:N, or mps.")
        if requested_device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable.")
            if requested_device.index is not None and requested_device.index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device index is unavailable: {device}")
        if requested_device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable.")

        self.name = name
        self.model_name = MODEL_IDS[name]
        self._torch = torch
        self.requested_revision = revision
        options = {}
        if name == "llama":
            from dotenv import load_dotenv

            load_dotenv()
            # Keep the existing alias without changing the user's global login.
            token = os.getenv("LLAMA_TOKEN")
            if token:
                options["token"] = token

        model_class = AutoModelForMaskedLM if name == "bert" else AutoModelForCausalLM
        self.model = model_class.from_pretrained(
            self.model_name, revision=revision, dtype=getattr(torch, dtype), **options,
        )
        self.model.to(requested_device)
        self.model.eval()
        self.model_revision = getattr(self.model.config, "_commit_hash", None)
        # Pin the tokenizer to the revision actually loaded for the model.
        tokenizer_revision = self.model_revision or revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=tokenizer_revision, **options,
        )
        self.tokenizer_revision = self.model_revision

    def run_metadata(self):
        """Describe inference without including authentication credentials."""
        return {
            "name": self.name, "model_id": self.model_name,
            "requested_revision": self.requested_revision,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "device": str(self.model.device), "dtype": str(self.model.dtype),
            "evaluation_mode": not self.model.training,
            "scoring_method": "masked_token_with_period" if self.name == "bert" else "causal_log_probability",
            "tokenization_policy": "masked_context_period_v1" if self.name == "bert" else "stripped_context_single_space_continuation_v1",
        }

    def tokenize_sentence(self, sentence: str):
        """Prepare a context; the candidate supplies the separating space."""
        if self.name == "bert":
            return sentence

        prompt = sentence.strip()
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True,
        ).to(self.model.device)
        return inputs if self.name == "deepseek" else inputs["input_ids"]

    def tokenize_word(self, word: str):
        """Tokenize a continuation with a leading space and no special tokens."""
        word = word.strip()
        if not word:
            return None
        return self.tokenizer.encode(" " + word, add_special_tokens=False) or None

    def predict_next_word(self, context, word_token_ids):
        """Return a natural log score, or None for unsupported BERT words."""
        with self._torch.no_grad():
            if self.name == "bert":
                return self._score_masked(context, word_token_ids)
            if self.name == "deepseek":
                return self._score_deepseek(context, word_token_ids)
            return self._score_causal(context, word_token_ids)

    def _score_masked(self, sentence, word_token_ids):
        if len(word_token_ids) != 1:
            return None
        inputs = self.tokenizer(
            sentence + " [MASK].", return_tensors="pt", add_special_tokens=True,
        ).to(self.model.device)
        logits = self.model(**inputs).logits
        mask_index = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        mask_logits = logits[0, mask_index, :].squeeze()
        log_probs = self._torch.nn.functional.log_softmax(mask_logits, dim=-1)
        return log_probs[word_token_ids[0]].item()

    def _score_deepseek(self, inputs, word_token_ids):
        # Rebind tensors in a local mapping so candidates can share a context.
        inputs = dict(inputs)
        total = 0.0
        for index, token_id in enumerate(word_token_ids):
            logits = self.model(**inputs, use_cache=False).logits[0, -1, :]
            log_probs = self._torch.nn.functional.log_softmax(logits, dim=-1)
            total += log_probs[token_id].item()

            if index + 1 < len(word_token_ids):
                input_ids = inputs["input_ids"]
                next_id = input_ids.new_tensor([[token_id]])
                inputs["input_ids"] = self._torch.cat([input_ids, next_id], dim=1)
                if "attention_mask" in inputs:
                    mask = inputs["attention_mask"]
                    inputs["attention_mask"] = self._torch.cat(
                        [mask, mask.new_ones((1, 1))], dim=1,
                    )
        return total

    def _score_causal(self, input_ids, word_token_ids):
        total = 0.0
        past_key_values = None
        for token_id in word_token_ids:
            if self.name == "llama":
                outputs = self.model(
                    input_ids=input_ids, past_key_values=past_key_values, use_cache=True,
                )
                past_key_values = outputs.past_key_values
            else:
                outputs = self.model(input_ids=input_ids, use_cache=False)

            logits = outputs.logits[0, -1, :]
            log_probs = self._torch.nn.functional.log_softmax(logits, dim=-1)
            total += log_probs[token_id].item()

            if self.name == "llama":
                input_ids = self._torch.tensor([[token_id]], device=input_ids.device)
            else:
                next_id = self._torch.tensor([[token_id]], device=self.model.device)
                input_ids = self._torch.cat([input_ids, next_id], dim=1)
        return total
