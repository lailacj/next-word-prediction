"""Combine the audited Peelle repeats using original response counts."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import tempfile

from .artifacts import publish, sha256, write_json
from .datasets import REQUIRED_COLUMNS, load_cloze_data
from .provenance import timestamp


def read_rows(path):
    load_cloze_data(path, cloze_scale='proportion')
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        return [{c: row[c] for c in REQUIRED_COLUMNS} for row in csv.DictReader(file)]


def candidate_key(row):
    return row['sentence_number'], row['word']


def canonical_word(word):
    """The two transformations observed in the audited imported repeats."""
    return word.strip().replace('-', ' ')


def combine_rows(rows, groups):
    """Require a complete count-based explanation of every repeated key."""
    indexed = defaultdict(list)
    for number, row in enumerate(rows, 1):
        indexed[candidate_key(row)].append((number, row))
    repeats = {key for key, values in indexed.items() if len(values) > 1}
    corrections = {}
    for group in groups:
        key = candidate_key(group)
        if key in corrections:
            raise ValueError(f'Duplicate correction for {key!r}')
        corrections[key] = group
    if repeats != corrections.keys():
        raise ValueError('Corrections must cover exactly the repeated candidate keys.')

    merged, audit = {}, []
    for key, group in corrections.items():
        originals = indexed[key]
        if any(row['sentence'] != group['sentence'] for _, row in originals):
            raise ValueError(f'Conflicting context for {key!r}')
        variants = group['variants']
        if len(variants) != len(originals) or len({v['word'] for v in variants}) != len(variants):
            raise ValueError(f'Expected distinct upstream variants for {key!r}')
        if any(canonical_word(v['word']) != key[1] for v in variants):
            raise ValueError(f'Upstream variant does not match {key!r}')
        for variant in variants:
            count, total = variant['count'], variant['total']
            if type(count) is not int or type(total) is not int or not 0 < count <= total:
                raise ValueError(f'Invalid response counts for {key!r}')
            if round(count / total, 2) != float(variant['published_cloze']):
                raise ValueError(f'Counts do not reproduce published cloze for {key!r}')
        totals = {v['total'] for v in variants}
        if len(totals) != 1:
            raise ValueError(f'Conflicting response denominators for {key!r}')
        if Counter(float(row['cloze_prob']) for _, row in originals) != Counter(
            float(v['published_cloze']) for v in variants
        ):
            raise ValueError(f'Published cloze does not match input for {key!r}')
        count, total = sum(v['count'] for v in variants), totals.pop()
        if count > total:
            raise ValueError(f'Combined count exceeds denominator for {key!r}')
        row = dict(originals[0][1], cloze_prob=str(count / total))
        merged[key] = row
        audit.append({
            **group, 'input_rows': [number for number, _ in originals],
            'original_cloze': [r['cloze_prob'] for _, r in originals],
            'combined_count': count, 'total': total, 'combined_cloze': count / total,
        })

    output, seen = [], set()
    for row in rows:
        key = candidate_key(row)
        if key not in seen:
            output.append(merged.get(key, row))
            seen.add(key)
    return output, audit


def combine_peelle(source, corrections, output, *, overwrite=False):
    source, corrections, output = map(Path, (source, corrections, output))
    manifest = output.with_suffix('.csv.combination.json')
    for path in (output, manifest):
        if path.resolve() in {source.resolve(), corrections.resolve()}:
            raise ValueError('Combination must not overwrite input files.')
        if path.exists() and not overwrite:
            raise FileExistsError(f'{path} exists; use --overwrite to replace it.')
    source_hash, corrections_hash = sha256(source), sha256(corrections)
    evidence = json.loads(corrections.read_text(encoding='utf-8'))
    if evidence['schema_version'] != 1 or evidence['input_sha256'] != source_hash:
        raise ValueError('Corrections do not match this input file or schema.')
    rows = read_rows(source)
    combined, audit = combine_rows(rows, evidence['groups'])
    if sha256(source) != source_hash or sha256(corrections) != corrections_hash:
        raise ValueError('Input files changed during combination.')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=output.parent, encoding='utf-8', newline='', delete=False) as file:
            temporary = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(combined)
        records = load_cloze_data(temporary, cloze_scale='proportion')
        report = {
            'schema_version': 1, 'dataset': 'peelle', 'combined_at': timestamp(),
            'source': {'path': str(source), 'sha256': source_hash, 'rows': len(rows)},
            'corrections': {'path': str(corrections), 'sha256': corrections_hash},
            'output': {'path': str(output), 'sha256': sha256(temporary),
                       'rows': len(combined), 'sentences': len(records), 'cloze_scale': 'proportion'},
            'merged_groups': len(audit), 'removed_rows': len(rows) - len(combined),
            'cloze_rule': 'sum original response counts / shared sentence response total',
            'other_rows': 'preserve original published probabilities; no renormalization',
            'groups': audit,
        }
        publish(temporary, output, overwrite=overwrite)
        write_json(manifest, report)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/processed/peelle.csv'))
    parser.add_argument('--corrections', type=Path, default=Path('data/corrections/peelle_repeats.json'))
    parser.add_argument('--output', type=Path, default=Path('data/analysis/peelle.csv'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    try:
        report = combine_peelle(args.input, args.corrections, args.output, overwrite=args.overwrite)
        print(f"Combined {report['merged_groups']} groups; wrote {report['output']['rows']} rows to {args.output}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
