"""Numerical checks with tiny random models; no pretrained assets are loaded.

Run explicitly with: python -m unittest discover -s tests/integration -v
Requires torch and transformers, unlike the stand-in tests in tests/.
"""

import unittest
import csv
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
from transformers import (
    BertConfig, BertForMaskedLM, LlamaConfig, LlamaForCausalLM,
    PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM,
)

from next_word_prediction.models import LanguageModel
from next_word_prediction.pipeline import run_pipeline


def tiny_model(name):
    config_class, model_class = (
        (LlamaConfig, LlamaForCausalLM) if name == "llama"
        else (Qwen2Config, Qwen2ForCausalLM)
    )
    config = config_class(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.3,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(1234)
        return model_class(config).eval()


@torch.no_grad()
def reference_score(model, prompt, candidate, mask=None):
    """Score all continuation positions together in one uncached forward pass."""
    targets = torch.tensor([candidate], dtype=prompt.dtype, device=prompt.device)
    sequence = torch.cat((prompt, targets), dim=1)
    full_mask = None if mask is None else torch.cat((mask, torch.ones_like(targets)), dim=1)
    logits = model(input_ids=sequence, attention_mask=full_mask, use_cache=False).logits
    # The last prompt position predicts the first target token.
    prediction_logits = logits[:, prompt.shape[1] - 1:-1, :]
    token_scores = prediction_logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1))
    return token_scores.double().sum().item()


class NumericalScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {name: tiny_model(name) for name in ("qwen", "deepseek", "llama")}

    def scorer(self, name):
        # Inject a local model to exercise the production scoring methods.
        wrapper = LanguageModel.__new__(LanguageModel)
        wrapper.name = name
        wrapper._torch = torch
        wrapper.model = self.models[name]
        return wrapper

    def test_causal_scores_match_single_forward_reference(self):
        for name in self.models:
            scorer = self.scorer(name)
            for tokens in ([1], [1, 5, 8]):
                for candidate in ([4], [4, 9], [9, 4, 7]):
                    with self.subTest(model=name, prompt=tokens, candidate=candidate):
                        prompt = torch.tensor([tokens])
                        mask = torch.ones_like(prompt)
                        context = {"input_ids": prompt, "attention_mask": mask} if name == "deepseek" else prompt
                        actual = scorer.predict_next_word(context, candidate)
                        expected = reference_score(scorer.model, prompt, candidate, mask if name == "deepseek" else None)
                        self.assertAlmostEqual(actual, expected, delta=1e-5)
                        self.assertTrue(all(p.grad is None for p in scorer.model.parameters()))

    def test_deepseek_mask_and_optional_mask_match_reference(self):
        scorer = self.scorer("deepseek")
        prompt = torch.tensor([[0, 1, 5]])
        for mask in (torch.tensor([[0, 1, 1]]), None):
            with self.subTest(mask=mask):
                context = {"input_ids": prompt}
                if mask is not None:
                    context["attention_mask"] = mask
                actual = scorer.predict_next_word(context, [4, 9, 7])
                self.assertAlmostEqual(actual, reference_score(scorer.model, prompt, [4, 9, 7], mask), delta=1e-5)

    def test_candidate_order_and_original_context_are_preserved(self):
        candidates = ([4, 9], [9, 7, 4], [7])
        for name in self.models:
            with self.subTest(model=name):
                scorer = self.scorer(name)
                prompt = torch.tensor([[1, 5, 8]])
                mask = torch.ones_like(prompt)
                original_prompt, original_mask = prompt.clone(), mask.clone()
                context = {"input_ids": prompt, "attention_mask": mask} if name == "deepseek" else prompt
                forward = {tuple(c): scorer.predict_next_word(context, c) for c in candidates}
                reverse = {tuple(c): scorer.predict_next_word(context, c) for c in reversed(candidates)}
                self.assertEqual(forward, reverse)
                torch.testing.assert_close(prompt, original_prompt, rtol=0, atol=0)
                torch.testing.assert_close(mask, original_mask, rtol=0, atol=0)
                if name == "deepseek":
                    self.assertIs(context["input_ids"], prompt)
                    self.assertIs(context["attention_mask"], mask)

    def test_constructor_disables_training_mode(self):
        model = tiny_model("qwen").train()
        self.assertTrue(model.training)
        with patch("transformers.AutoTokenizer.from_pretrained"), patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=model,
        ):
            wrapper = LanguageModel("qwen")
        self.assertFalse(wrapper.model.training)
        prompt = torch.tensor([[1, 5, 8]])
        self.assertEqual(wrapper.predict_next_word(prompt, [4, 9]),
                         wrapper.predict_next_word(prompt, [4, 9]))

    def test_bert_masked_score_with_local_wordpiece_tokenizer(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, processors

        vocab = {word: i for i, word in enumerate(
            ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "A", "context", ".", "answer"),
        )}
        backend = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        backend.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
        )
        wrapper = LanguageModel.__new__(LanguageModel)
        wrapper.name, wrapper._torch = "bert", torch
        wrapper.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", mask_token="[MASK]",
            cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]",
        )
        with torch.random.fork_rng():
            torch.manual_seed(1234)
            wrapper.model = BertForMaskedLM(BertConfig(
                vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=32,
            )).eval()
        context = wrapper.tokenize_sentence("A context")
        candidate = wrapper.tokenize_word("answer")
        self.assertEqual(candidate, [8])
        actual = wrapper.predict_next_word(context, candidate)
        # Explicit IDs specify [CLS] A context [MASK] . [SEP], independently of
        # the production prompt preparation and mask-position lookup.
        ids = torch.tensor([[2, 5, 6, 4, 7, 3]])
        with torch.no_grad():
            logits = wrapper.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
            expected = logits[0, 3].log_softmax(-1)[8].item()
        self.assertAlmostEqual(actual, expected, delta=1e-5)
        self.assertIsNone(wrapper.predict_next_word(context, [8, 8]))
        # Exercise the complete CSV -> real tiny model -> CSV + metadata path.
        wrapper.model_name = "test/tiny-bert"
        wrapper.requested_revision = "local-random-test"
        wrapper.model_revision = wrapper.tokenizer_revision = None
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "peelle.csv"
            dataset.write_text("sentence_number,sentence,word,cloze_prob\n1,A context,answer,0.5\n")
            output = Path(directory) / "bert.csv"
            with patch("next_word_prediction.pipeline.create_model", return_value=wrapper):
                self.assertEqual(run_pipeline(dataset, "bert", output), (1, 0))
            with output.open() as file:
                row = next(csv.DictReader(file))
            self.assertAlmostEqual(float(row["bert_prob"]), expected, delta=1e-5)
            metadata = json.loads(output.with_suffix(".csv.metadata.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["model"]["dtype"], "torch.float32")
            self.assertEqual(metadata["model"]["scoring_method"], "masked_token_with_period")



if __name__ == "__main__":
    unittest.main()
