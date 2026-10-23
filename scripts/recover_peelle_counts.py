"""Rebuild the repeat corrections from a pinned upstream checkout (no models needed)."""

import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import string

from next_word_prediction.artifacts import sha256
from next_word_prediction.combine_peelle import candidate_key, canonical_word, combine_rows, read_rows

REVISION = '6bdd4426c35a76b2b1eecbe904d7277d061ea134'
REPOSITORY = 'https://github.com/jpeelle/sentence-prediction'
# Verify every input, including the upstream Python code before executing it.
HASHES = {
    'predict_sent_analysis.py': '3ea2358b1e44e476d5558413e2f7214425c431e62ff24dedd95083b841d3b8b5',
    'example/replacements.csv': '894a9490dc27cd2c437c4de063d6c991085d7436c8829a58d4b9346dc93b64c1',
    'example/anonymized_fill-in-1x35.csv': '32aca9a16bc7b97f7373080da104bf2cd09f45bf7abd577bbc0c11ac77473b75',
    'example/anonymized_fill-in-1x50.csv': '498aa70008c830f557339301c3d837b94c7e2713dd7ddb3986f35eac2a8cb6e8',
    'example/anonymized_fill-in-61x50.csv': 'a092bcc9100132f476f4fec9bc43ffa8a7869f4c2f84de4e5ff09d02040c9209',
    'example/censors.csv': '882c5409fa7789032a14e6e812ab3521dc686155660346fffa5fb9335abefef4',
    'example/excluded_workers.txt': '4e7e0d6fa20ca128074d1d965e32025191c204867293760b797e007e2c8aaf63',
}


def recover(upstream, source):
    upstream, source = Path(upstream), Path(source)
    for relative, expected in HASHES.items():
        if sha256(upstream / relative) != expected:
            raise ValueError(f'Unexpected upstream content: {relative}; expected revision {REVISION}')
    script = upstream / 'predict_sent_analysis.py'
    # Replay only the two reviewed counting functions, with their original parsing,
    # corrections, exclusions, and denominator rules. Do not run the upstream CLI.
    tree = ast.parse(script.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {'csv_reader', 'word_replacer'}]
    namespace = {'string': string}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(script), 'exec'), namespace)
    example = upstream / 'example'
    questions, _ = namespace['csv_reader'](
        [example / f'anonymized_fill-in-{part}.csv' for part in ('1x35', '1x50', '61x50')],
        example / 'replacements.csv', example / 'censors.csv', example / 'excluded_workers.txt',
    )
    questions = list(questions.items())
    rows = read_rows(source)
    indexed = defaultdict(list)
    for row in rows:
        indexed[candidate_key(row)].append(row)
    groups = []
    for (sentence_id, word), originals in indexed.items():
        if len(originals) == 1:
            continue
        question, answers = questions[int(sentence_id) - 1]
        if question.removesuffix(' __________.') != originals[0]['sentence']:
            raise ValueError(f'Upstream sentence mismatch for {sentence_id}')
        variants = []
        for raw_word, (published_cloze, fraction) in answers.items():
            if canonical_word(raw_word) == word:
                count, total = map(int, fraction.split('/'))
                variants.append({'word': raw_word, 'count': count, 'total': total,
                                 'published_cloze': published_cloze})
        if Counter(float(r['cloze_prob']) for r in originals) != Counter(v['published_cloze'] for v in variants):
            raise ValueError(f'Upstream probabilities mismatch for {(sentence_id, word)}')
        groups.append({
            'sentence_number': sentence_id, 'sentence': originals[0]['sentence'], 'word': word,
            'cause': 'hyphen_to_space' if any('-' in v['word'] for v in variants) else 'boundary_whitespace',
            'variants': variants,
        })
    combine_rows(rows, groups)  # Check counts, denominators, and complete coverage.
    return {
        'schema_version': 1, 'input_sha256': sha256(source),
        'upstream': {'repository': REPOSITORY, 'revision': REVISION, 'sha256': HASHES},
        'recovery': 'scripts/recover_peelle_counts.py; original upstream csv_reader and word_replacer',
        'canonicalization': 'strip boundary whitespace; replace hyphens with spaces; repeated keys only',
        'groups': groups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-dir', required=True, type=Path)
    parser.add_argument('--input', type=Path, default=Path('data/processed/peelle.csv'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = recover(args.upstream_dir, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(evidence, file, indent=2, allow_nan=False)
        file.write('\n')
    print(f"Recovered counts for {len(evidence['groups'])} repeated keys: {args.output}")


if __name__ == '__main__':
    main()
