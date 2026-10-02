"""Check model dispatch and scoring conventions without downloading weights."""

import os
import unittest
from unittest.mock import MagicMock, patch

from next_word_prediction.models import LanguageModel, MODEL_IDS
from next_word_prediction import pipeline as runner
from next_word_prediction import cli


class LanguageModelTests(unittest.TestCase):
    def setUp(self):
        self.torch = MagicMock()
        self.transformers = MagicMock()
        self.hub = MagicMock()
        self.dotenv = MagicMock()
        modules = {
            "torch": self.torch,
            "transformers": self.transformers,
            "huggingface_hub": self.hub,
            "dotenv": self.dotenv,
        }
        patcher = patch.dict("sys.modules", modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_loading_selects_checkpoint_and_model_type(self):
        for name, checkpoint in MODEL_IDS.items():
            with self.subTest(name=name):
                self.transformers.reset_mock()
                self.hub.reset_mock()
                self.dotenv.reset_mock()
                with patch.dict(os.environ, {"LLAMA_TOKEN": "test-token"}):
                    model = LanguageModel(name)
                options = {"trust_remote_code": True} if name == "deepseek" else {}
                self.transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
                    checkpoint, **options,
                )
                selected = (self.transformers.AutoModelForMaskedLM if name == "bert"
                            else self.transformers.AutoModelForCausalLM)
                other = (self.transformers.AutoModelForCausalLM if name == "bert"
                         else self.transformers.AutoModelForMaskedLM)
                selected.from_pretrained.assert_called_once_with(checkpoint, **options)
                other.from_pretrained.assert_not_called()
                self.assertEqual(model.model_name, checkpoint)
                if name == "llama":
                    self.dotenv.load_dotenv.assert_called_once_with()
                    self.hub.login.assert_called_once_with(token="test-token")
                else:
                    self.dotenv.load_dotenv.assert_not_called()
                    self.hub.login.assert_not_called()

    def test_unknown_model_fails_before_loading(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            LanguageModel("unknown")
        self.transformers.AutoTokenizer.from_pretrained.assert_not_called()

    def test_sentence_tokenization_preserves_model_conventions(self):
        for name in MODEL_IDS:
            with self.subTest(name=name):
                model = LanguageModel(name)
                model.tokenizer = MagicMock()
                inputs = model.tokenizer.return_value
                inputs.to.return_value = inputs
                result = model.tokenize_sentence("  Some context  ")
                if name == "bert":
                    self.assertEqual(result, "  Some context  ")
                    model.tokenizer.assert_not_called()
                else:
                    prompt = "Some context " if name == "qwen" else "Some context"
                    model.tokenizer.assert_called_once_with(prompt, return_tensors="pt")
                    if name in ("qwen", "deepseek"):
                        inputs.to.assert_called_once_with(model.model.device)
                    else:
                        inputs.to.assert_not_called()
                    self.assertIs(result, inputs if name == "deepseek" else inputs["input_ids"])

    def test_word_tokenization_is_shared(self):
        for name in MODEL_IDS:
            with self.subTest(name=name):
                model = LanguageModel(name)
                model.tokenizer.encode.reset_mock()
                model.tokenizer.encode.return_value = [4, 9]
                self.assertEqual(model.tokenize_word(" word "), [4, 9])
                model.tokenizer.encode.assert_called_once_with(" word", add_special_tokens=False)
                model.tokenizer.encode.return_value = []
                self.assertIsNone(model.tokenize_word(""))

    def prepare_scores(self, model, values):
        distributions = []
        for value in values:
            distribution = MagicMock()
            distribution.__getitem__.return_value.item.return_value = value
            distributions.append(distribution)
        self.torch.nn.functional.log_softmax.side_effect = distributions
        model.model.reset_mock()
        return distributions

    def test_qwen_extends_full_context_and_sums_token_scores(self):
        model = LanguageModel("qwen")
        scores = self.prepare_scores(model, [-0.25, -0.75])
        context = MagicMock()
        self.assertEqual(model.predict_next_word(context, [4, 9]), -1.0)
        calls = model.model.call_args_list
        self.assertIs(calls[0].kwargs["input_ids"], context)
        self.assertIs(calls[1].kwargs["input_ids"], self.torch.cat.return_value)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.torch.cat.call_args_list[0].args[0], [context, self.torch.tensor.return_value])
        scores[0].__getitem__.assert_called_once_with(4)
        scores[1].__getitem__.assert_called_once_with(9)

    def test_llama_passes_cache_and_only_new_token(self):
        model = LanguageModel("llama")
        self.prepare_scores(model, [-0.25, -0.75])
        context = MagicMock()
        self.assertEqual(model.predict_next_word(context, [4, 9]), -1.0)
        calls = model.model.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0].kwargs["input_ids"], context)
        self.assertIsNone(calls[0].kwargs["past_key_values"])
        self.assertTrue(calls[0].kwargs["use_cache"])
        self.assertIs(calls[1].kwargs["input_ids"], self.torch.tensor.return_value)
        self.assertIs(calls[1].kwargs["past_key_values"], model.model.return_value.past_key_values)
        self.torch.cat.assert_not_called()

    def test_bert_scores_mask_and_skips_multiple_tokens(self):
        model = LanguageModel("bert")
        scores = self.prepare_scores(model, [-0.25])
        model.tokenizer.reset_mock()
        inputs = model.tokenizer.return_value
        inputs.input_ids.__eq__.return_value = MagicMock()
        self.assertIsNone(model.predict_next_word("A context", [4, 9]))
        model.model.assert_not_called()
        model.tokenizer.assert_not_called()
        self.assertEqual(model.predict_next_word("A context", [4]), -0.25)
        model.tokenizer.assert_called_once_with("A context [MASK].", return_tensors="pt")
        scores[0].__getitem__.assert_called_once_with(4)
        inputs.input_ids.__eq__.assert_called_once_with(model.tokenizer.mask_token_id)

    def test_runner_constructs_unified_model(self):
        with patch.object(runner, "LanguageModel") as model_class:
            self.assertIs(runner.create_model("qwen"), model_class.return_value)
            model_class.assert_called_once_with("qwen")
        self.assertEqual(cli.MODEL_NAMES, tuple(MODEL_IDS))


if __name__ == "__main__":
    unittest.main()
