"""Exercise context-dependent scores without downloading model weights."""

from contextlib import nullcontext
import math
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from next_word_prediction.models import LanguageModel


class Tensor:
    """Minimal immutable, single-row tensor for checking token/mask contents."""

    def __init__(self, values, *, dtype="int64", device="test-device"):
        self.values = tuple(values)
        self.dtype = dtype
        self.device = device

    def new_tensor(self, rows):
        assert len(rows) == 1
        return Tensor(rows[0], dtype=self.dtype, device=self.device)

    def new_ones(self, shape):
        assert shape == (1, 1)
        return self.new_tensor([[1]])


def concatenate(tensors, dim):
    assert dim == 1
    first = tensors[0]
    assert all(t.dtype == first.dtype and t.device == first.device for t in tensors)
    return Tensor([value for t in tensors for value in t.values],
                  dtype=first.dtype, device=first.device)


class DeepSeekScoringTests(unittest.TestCase):
    def setUp(self):
        # These distributions deliberately differ when a target token is appended.
        self.probabilities = {
            (0, 2): {4: .2, 9: .1, 7: .1},
            (0, 2, 4): {9: .6},
            (0, 2, 4, 9): {7: .8},
            (0, 2, 9): {7: .3},
        }
        self.model = LanguageModel.__new__(LanguageModel)
        self.model.name = "deepseek"
        self.model._torch = MagicMock()
        self.model._torch.no_grad.side_effect = nullcontext
        self.model._torch.cat.side_effect = concatenate
        # The forward stub supplies the known distribution for each actual context.
        self.model._torch.nn.functional.log_softmax.side_effect = lambda logits, dim: logits
        self.model.model = MagicMock(side_effect=self.forward)
        self.context = {"input_ids": Tensor([0, 2]), "attention_mask": Tensor([0, 1], dtype="bool")}

    def forward(self, input_ids, attention_mask=None, use_cache=True):
        self.assertFalse(use_cache)
        if attention_mask is not None:
            self.assertEqual(len(input_ids.values), len(attention_mask.values))
        distribution = MagicMock()
        distribution.__getitem__.side_effect = lambda token: SimpleNamespace(
            item=lambda: math.log(self.probabilities[input_ids.values][token]),
        )
        logits = MagicMock()
        logits.__getitem__.return_value = distribution
        return SimpleNamespace(logits=logits)

    def test_multitoken_score_uses_growing_context_and_mask(self):
        score = self.model.predict_next_word(self.context, [4, 9, 7])
        self.assertAlmostEqual(score, math.log(.2) + math.log(.6) + math.log(.8))
        calls = self.model.model.call_args_list
        self.assertEqual([c.kwargs["input_ids"].values for c in calls],
                         [(0, 2), (0, 2, 4), (0, 2, 4, 9)])
        self.assertEqual([c.kwargs["attention_mask"].values for c in calls],
                         [(0, 1), (0, 1, 1), (0, 1, 1, 1)])
        for call in calls:
            self.assertEqual(call.kwargs["input_ids"].dtype, "int64")
            self.assertEqual(call.kwargs["attention_mask"].dtype, "bool")
            self.assertEqual(call.kwargs["input_ids"].device, "test-device")

    def test_candidates_do_not_change_shared_context(self):
        original = dict(self.context)
        self.model.predict_next_word(self.context, [4, 9, 7])
        score = self.model.predict_next_word(self.context, [9, 7])
        self.assertAlmostEqual(score, math.log(.1) + math.log(.3))
        self.assertEqual(self.context, original)
        self.assertEqual(self.context["input_ids"].values, (0, 2))
        self.assertEqual(self.context["attention_mask"].values, (0, 1))
        again = self.model.predict_next_word(self.context, [4, 9, 7])
        self.assertAlmostEqual(again, math.log(.2) + math.log(.6) + math.log(.8))

    def test_single_token_keeps_original_score_and_inputs(self):
        self.assertAlmostEqual(self.model.predict_next_word(self.context, [9]), math.log(.1))
        self.model.model.assert_called_once_with(**self.context, use_cache=False)
        self.model._torch.cat.assert_not_called()

    def test_context_without_attention_mask(self):
        context = {"input_ids": self.context["input_ids"]}
        self.assertAlmostEqual(self.model.predict_next_word(context, [4, 9]), math.log(.2) + math.log(.6))
        self.assertTrue(all("attention_mask" not in c.kwargs for c in self.model.model.call_args_list))
        self.assertEqual(context["input_ids"].values, (0, 2))


if __name__ == "__main__":
    unittest.main()
