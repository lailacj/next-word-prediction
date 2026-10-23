"""Combining response categories must preserve counts and scored inputs."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from next_word_prediction.artifacts import sha256
from next_word_prediction.combine_peelle import combine_peelle, combine_rows, read_rows


class PeelleCombinationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / 'source.csv'
        self.source.write_text(
            'sentence_number,sentence,word,cloze_prob\n'
            '009,A blue gem,sapphire,0.50\n'
            '009,A blue gem,diamond,0.20\n'
            '009,A blue gem,sapphire,0.05\n'
            '10,Hold it with a,safety pin,0.01\n'
            '10,Hold it with a,safety pin,0.01\n'
        )
        self.groups = [
            {'sentence_number': '009', 'sentence': 'A blue gem', 'word': 'sapphire',
             'variants': [{'word': 'sapphire', 'count': 53, 'total': 105, 'published_cloze': .50},
                          {'word': 'sapphire ', 'count': 5, 'total': 105, 'published_cloze': .05}]},
            {'sentence_number': '10', 'sentence': 'Hold it with a', 'word': 'safety pin',
             'variants': [{'word': 'safety-pin', 'count': 1, 'total': 104, 'published_cloze': .01},
                          {'word': 'safety pin', 'count': 1, 'total': 104, 'published_cloze': .01}]},
        ]
        self.corrections = self.root / 'corrections.json'
        self.corrections.write_text(json.dumps({
            'schema_version': 1, 'input_sha256': sha256(self.source), 'groups': self.groups,
        }))
        self.output = self.root / 'analysis' / 'peelle.csv'

    def test_count_pooling_equal_probabilities_order_and_provenance(self):
        original = self.source.read_bytes()
        report = combine_peelle(self.source, self.corrections, self.output)
        rows = read_rows(self.output)
        self.assertEqual([r['word'] for r in rows], ['sapphire', 'diamond', 'safety pin'])
        self.assertEqual(rows[0]['sentence_number'], '009')
        self.assertEqual(float(rows[0]['cloze_prob']), 58 / 105)
        self.assertNotEqual(float(rows[0]['cloze_prob']), .55)
        self.assertEqual(rows[1]['cloze_prob'], '0.20')
        self.assertEqual(float(rows[2]['cloze_prob']), 2 / 104)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(report['removed_rows'], 2)
        self.assertEqual(report['output']['sha256'], sha256(self.output))
        audit = json.loads(self.output.with_suffix('.csv.combination.json').read_text())
        self.assertEqual(audit['groups'][0]['input_rows'], [1, 3])
        self.assertEqual(audit['groups'][0]['combined_count'], 58)

    def test_all_and_only_repeated_keys_must_be_explained(self):
        rows = read_rows(self.source)
        for groups in (self.groups[:1], self.groups + [self.groups[0]]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                combine_rows(rows, groups)

    def test_bad_evidence_is_rejected(self):
        for field, value, message in (
            ('word', 'another word', 'does not match'),
            ('count', 0, 'Invalid response counts'),
            ('count', 52, 'reproduce published cloze'),
            ('total', 104, 'Conflicting response denominators'),
            ('published_cloze', .04, 'reproduce published cloze'),
        ):
            with self.subTest(field=field, value=value):
                groups = deepcopy(self.groups)
                groups[0]['variants'][1][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    combine_rows(read_rows(self.source), groups)

    def test_context_and_input_probability_mismatches_are_rejected(self):
        for field, value, message in (
            ('sentence', 'Different context', 'Conflicting context'),
            ('cloze_prob', '0.04', 'does not match input'),
        ):
            rows = read_rows(self.source)
            rows[2][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                combine_rows(rows, self.groups)

    def test_changed_input_is_rejected_before_publication(self):
        with self.source.open('a') as file:
            file.write('11,Another context,word,0.2\n')
        with self.assertRaisesRegex(ValueError, 'do not match this input'):
            combine_peelle(self.source, self.corrections, self.output)
        self.assertFalse(self.output.exists())

    def test_outputs_and_sources_are_protected(self):
        combine_peelle(self.source, self.corrections, self.output)
        original = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            combine_peelle(self.source, self.corrections, self.output)
        with self.assertRaisesRegex(ValueError, 'overwrite input files'):
            combine_peelle(self.source, self.corrections, self.source, overwrite=True)
        self.corrections.write_text('{}')
        with self.assertRaises(KeyError):
            combine_peelle(self.source, self.corrections, self.output, overwrite=True)
        self.assertEqual(self.output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
