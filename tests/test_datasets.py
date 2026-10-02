"""Validation and normalization checks using small CSV fixtures."""

import csv
from pathlib import Path
import tempfile
import unittest

from next_word_prediction.datasets import Candidate, SentenceRecord, load_cloze_data

HEADER = ["sentence_number", "sentence", "word", "cloze_prob"]


class DatasetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def write_csv(self, rows, name="michaelov_2024.csv", header=HEADER):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_registered_scales_and_boundaries(self):
        for name, raw, normalized in [
            ("michaelov_2024.csv", "0.2258064516129032", 0.2258064516129032),
            ("szewczyk_2022.csv", "0.182194616977226", 0.182194616977226),
            ("peelle.csv", "0.43", 0.43),
            ("nieuwland_2018.csv", "90", 0.9),
        ]:
            with self.subTest(name=name):
                top = "100" if name == "nieuwland_2018.csv" else "1"
                path = self.write_csv([["1", "Context", "a", raw], ["1", "Context", "b", "0"],
                                       ["1", "Context", "c", top]], name=name)
                self.assertEqual([c.cloze_prob for c in load_cloze_data(path)[0].candidates],
                                 [normalized, 0.0, 1.0])

    def test_unknown_filename_requires_scale_and_explicit_override_wins(self):
        path = self.write_csv([["1", "Context", "word", "1"]], name="custom.csv")
        with self.assertRaisesRegex(ValueError, "--cloze-scale"):
            load_cloze_data(path)
        self.assertEqual(load_cloze_data(path, cloze_scale="percent")[0].candidates[0].cloze_prob, 0.01)
        self.assertEqual(load_cloze_data(path, cloze_scale="proportion")[0].candidates[0].cloze_prob, 1.0)
        path = self.write_csv([["1", "Context", "word", "0.9"]], name="nieuwland_2018.csv")
        self.assertEqual(load_cloze_data(path, cloze_scale="proportion")[0].candidates[0].cloze_prob, 0.9)
        with self.assertRaises(ValueError):
            load_cloze_data(path, cloze_scale="guess")

    def test_preserves_text_ids_candidate_order_and_source_bytes(self):
        sentence = '\"Quoted, context\"\nwith a final apostrophe\''
        path = self.write_csv([["009", sentence, " first ", ".5"],
                               ["item-A", "Other context", "NA", "0"],
                               ["009", sentence, "second", ".25"]])
        original = path.read_bytes()
        self.assertEqual(load_cloze_data(path), [
            SentenceRecord("009", sentence, (Candidate(" first ", .5), Candidate("second", .25))),
            SentenceRecord("item-A", "Other context", (Candidate("NA", 0.0),)),
        ])
        self.assertEqual(path.read_bytes(), original)

    def test_rejects_conflicting_contexts(self):
        path = self.write_csv([["7", "First", "a", "0"], ["7", "Second", "b", "0"]])
        with self.assertRaisesRegex(ValueError, "line 3:.*conflicting sentence"):
            load_cloze_data(path)

    def test_rejects_invalid_cloze_values(self):
        for value in ("nan", "inf", "-inf", "-0.01", "1.01", "not numeric", ""):
            with self.subTest(value=value):
                path = self.write_csv([["1", "Context", "word", value]])
                with self.assertRaisesRegex(ValueError, "line 2: cloze_prob"):
                    load_cloze_data(path)
        path = self.write_csv([["1", "Context", "word", "100.01"]], name="nieuwland_2018.csv")
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            load_cloze_data(path)

    def test_rejects_blank_required_text(self):
        for index in range(3):
            with self.subTest(column=HEADER[index]):
                row = ["1", "Context", "word", "0.5"]
                row[index] = "  "
                path = self.write_csv([row])
                with self.assertRaisesRegex(ValueError, f"line 2: {HEADER[index]} must not be blank"):
                    load_cloze_data(path)

    def test_rejects_missing_duplicate_and_empty_headers(self):
        for header in (HEADER[:-1], HEADER + ["word"], HEADER + [""]):
            with self.subTest(header=header):
                path = self.write_csv([], header=header)
                with self.assertRaisesRegex(ValueError, "line 1:"):
                    load_cloze_data(path)

    def test_rejects_wrong_row_width_and_broken_csv(self):
        for row in (["1", "Context", "word"], ["1", "Context", "word", "0", "extra"]):
            path = self.write_csv([row])
            with self.assertRaisesRegex(ValueError, "line 2:.*number of fields"):
                load_cloze_data(path)
        path.write_text(','.join(HEADER) + '\n1,"unclosed,word,0.5\n')
        with self.assertRaisesRegex(ValueError, "invalid CSV"):
            load_cloze_data(path)

    def test_rejects_empty_inputs(self):
        path = self.write_csv([])
        with self.assertRaisesRegex(ValueError, "no candidate rows"):
            load_cloze_data(path)
        path.write_text("")
        with self.assertRaisesRegex(ValueError, "missing CSV header"):
            load_cloze_data(path)

    def test_accepts_bom_and_extra_named_columns(self):
        path = self.write_csv([["1", "Context", "word", "0.5", "metadata"]], header=HEADER + ["note"])
        path.write_bytes(b'\xef\xbb\xbf' + path.read_bytes())
        self.assertEqual(load_cloze_data(path)[0].candidates, (Candidate("word", 0.5),))


if __name__ == "__main__":
    unittest.main()
