"""Runner tests that require no model downloads or ML dependencies."""

import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from next_word_prediction import pipeline as runner
from next_word_prediction import cli


class FakeModel:
    def tokenize_sentence(self, sentence):
        return sentence

    def tokenize_word(self, word):
        return [] if word == "empty" else [word]

    def predict_next_word(self, sentence, tokens):
        return None if tokens == ["unsupported"] else -1.25


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "michaelov_2024.csv"
        self.dataset.write_text("sentence_number,sentence,word,cloze_prob\n1,A context,word,0.5\n", encoding="utf-8")
        self.output = self.root / "results" / "scores.csv"

    def test_scores_preserve_ids_and_csv_text_and_report_skips(self):
        sentence = 'She said, "hello"\nand waited for'
        with self.dataset.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["sentence_number", "sentence", "word", "cloze_prob"])
            writer.writerows([
                ["42", sentence, "reply", "0.5"],
                ["42", sentence, "empty", "0"],
                ["42", sentence, "unsupported", "0"],
                ["7", "Another context", "answer", "0.2"],
            ])
        with patch.object(runner, "create_model", return_value=FakeModel()):
            self.assertEqual(runner.run_pipeline(self.dataset, "bert", self.output), (2, 2))
        with self.output.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual([row["sentence_num"] for row in rows], ["42", "7"])
        self.assertEqual(rows[0]["sentence"], sentence)
        self.assertEqual(rows[0]["bert_prob"], "-1.25")

    def test_existing_output_is_preserved_before_loading_model(self):
        self.output.parent.mkdir()
        self.output.write_text("previous result", encoding="utf-8")
        with patch.object(runner, "create_model") as factory:
            with self.assertRaises(FileExistsError):
                runner.run_pipeline(self.dataset, "qwen", self.output)
            factory.assert_not_called()
        self.assertEqual(self.output.read_text(), "previous result")

    def test_overwrite_replaces_output(self):
        self.output.parent.mkdir()
        self.output.write_text("previous result", encoding="utf-8")
        with patch.object(runner, "create_model", return_value=FakeModel()):
            runner.run_pipeline(self.dataset, "qwen", self.output, overwrite=True)
        self.assertEqual(self.output.read_text(), "sentence_num,sentence,word,qwen_prob\n1,A context,word,-1.25\n")

    def test_input_cannot_be_overwritten(self):
        with self.assertRaises(ValueError):
            runner.run_pipeline(self.dataset, "qwen", self.dataset, overwrite=True)
        self.assertIn("1,A context,word,0.5", self.dataset.read_text())

    def test_missing_input_fails_before_loading_model(self):
        with patch.object(runner, "create_model") as factory:
            with self.assertRaises(FileNotFoundError):
                runner.run_pipeline(self.root / "missing.csv", "qwen", self.output)
            factory.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_invalid_dataset_fails_before_model_or_output_changes(self):
        self.dataset.write_text("sentence_number,sentence,word,cloze_prob\n1,Context,word,nan\n")
        self.output.parent.mkdir()
        self.output.write_text("existing results")
        with patch.object(runner, "create_model") as factory:
            with self.assertRaisesRegex(ValueError, "line 2: cloze_prob"):
                runner.run_pipeline(self.dataset, "qwen", self.output, overwrite=True)
            factory.assert_not_called()
        self.assertEqual(self.output.read_text(), "existing results")

    def test_cli_forwards_explicit_scale(self):
        with patch.object(cli, "run_pipeline", return_value=(1, 0)) as run:
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(["--dataset", str(self.dataset), "--model", "bert", "--cloze-scale", "percent"])
        self.assertEqual(run.call_args.kwargs["cloze_scale"], "percent")

    def test_cli_selects_model_and_default_or_explicit_output(self):
        for model in cli.MODEL_NAMES:
            for custom in (False, True):
                with self.subTest(model=model, custom=custom):
                    argv = ["--dataset", str(self.dataset), "--model", model]
                    expected = Path.cwd() / "results" / "michaelov_2024" / f"{model}.csv"
                    if custom:
                        argv += ["--output", str(self.output), "--overwrite"]
                        expected = self.output
                    with patch.object(cli, "run_pipeline", return_value=(3, 1)) as run:
                        with contextlib.redirect_stdout(io.StringIO()):
                            cli.main(argv)
                    run.assert_called_once_with(self.dataset, model, expected, overwrite=custom, cloze_scale=None)


if __name__ == "__main__":
    unittest.main()
