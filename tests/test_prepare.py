"""Source preparation must preserve candidates and prevent target leakage."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from next_word_prediction.datasets import load_cloze_data
from next_word_prediction.prepare import PROFILES, prepare_dataset


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root / 'prepared'

    def source(self, name, rows):
        relative, column, _ = PROFILES[name]
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerow(['FullText', 'TargetWords', column])
            writer.writerows(rows)
        return path

    def test_tsv_profiles_deduplicate_measurements_preserve_alternatives(self):
        for name in ('michaelov_2024', 'nieuwland_2018', 'szewczyk_2022'):
            with self.subTest(name=name):
                value = '50' if name == 'nieuwland_2018' else '0.5'
                source = self.source(name, [['A repeated word word', 'word', value],
                                            ['A repeated word word', 'word', value],
                                            ['A repeated word reply', 'reply', '0']])
                original = source.read_bytes()
                report = prepare_dataset(name, self.root, self.output)
                records = load_cloze_data(self.output / (name + '.csv'))
                self.assertEqual([r.sentence_id for r in records], ['1', '2'])
                self.assertEqual([r.sentence for r in records], ['A repeated word'] * 2)
                self.assertEqual([r.candidates[0].word for r in records], ['word', 'reply'])
                self.assertEqual(records[0].candidates[0].cloze_prob, .5)
                self.assertEqual(report['removed_repeated_measurements'], 1)
                self.assertEqual(source.read_bytes(), original)
                manifest = json.loads((self.output / (name + '.csv.preparation.json')).read_text())
                self.assertEqual(manifest['source']['rows'], 3)
                self.assertEqual(manifest['output']['candidates'], 2)

    def test_conflicting_cloze_values_fail_before_output(self):
        self.source('michaelov_2024', [['A word', 'word', '.5'], ['A word', 'word', '.6']])
        with self.assertRaisesRegex(ValueError, 'conflicting cloze'):
            prepare_dataset('michaelov_2024', self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_nonterminal_target_is_rejected(self):
        self.source('michaelov_2024', [['A word here', 'word', '.5']])
        with self.assertRaisesRegex(ValueError, 'must end'):
            prepare_dataset('michaelov_2024', self.root, self.output)

    def test_bad_probability_is_rejected(self):
        self.source('nieuwland_2018', [['A word', 'word', '101']])
        with self.assertRaisesRegex(ValueError, 'outside'):
            prepare_dataset('nieuwland_2018', self.root, self.output)

    def test_peelle_keeps_ids_and_rows(self):
        source = self.root / PROFILES['peelle'][0]
        source.parent.mkdir(parents=True)
        source.write_text('sentence_number,sentence,word,cloze_prob\n009,A context,word,0.5\n009,A context,other,0.2\n')
        prepare_dataset('peelle', self.root, self.output)
        self.assertEqual(load_cloze_data(source, cloze_scale='proportion'),
                         load_cloze_data(self.output / 'peelle.csv'))

    def test_overwrite_is_explicit_and_invalid_source_keeps_previous_output(self):
        self.source('michaelov_2024', [['A word', 'word', '.5']])
        prepare_dataset('michaelov_2024', self.root, self.output)
        output = self.output / 'michaelov_2024.csv'
        old = output.read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_dataset('michaelov_2024', self.root, self.output)
        self.source('michaelov_2024', [['A different ending', 'word', '.5']])
        with self.assertRaises(ValueError):
            prepare_dataset('michaelov_2024', self.root, self.output, overwrite=True)
        self.assertEqual(output.read_bytes(), old)


if __name__ == '__main__':
    unittest.main()
