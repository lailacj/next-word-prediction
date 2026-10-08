"""Prepare the four supported source datasets for the scoring pipeline."""

import argparse
import csv
import math
from pathlib import Path
import tempfile

from .artifacts import publish, sha256, write_json
from .datasets import CLOZE_SCALES, REQUIRED_COLUMNS, load_cloze_data
from .provenance import timestamp

PROFILES = {
    'michaelov_2024': ('james-michaelov_data/datasets/michaelov_2024.tsv', 'Cloze', 'proportion'),
    'nieuwland_2018': ('james-michaelov_data/datasets/nieuwland_2018.tsv', 'cloze', 'percent'),
    'szewczyk_2022': ('james-michaelov_data/datasets/szewczyk_2022.tsv', 'cloze_p', 'proportion'),
    'peelle': ('peelle_data/cloze_data.csv', None, 'proportion'),
}


def convert_tsv(source, cloze_column, scale):
    """Collapse repeated measurements, preserving distinct candidate words."""
    candidates, output = {}, []
    count = 0
    with Path(source).open(encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file, delimiter='\t', strict=True)
        fields = reader.fieldnames or []
        required = {'FullText', 'TargetWords', cloze_column}
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError(f'{source}: missing or duplicate columns; required: {sorted(required)}')
        try:
            for row in reader:
                count += 1
                prefix = f'{source}: line {reader.line_num}'
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f'{prefix}: row width differs from header')
                text, word = row['FullText'].strip(), row['TargetWords'].strip()
                # These profiles define FullText as context + space + target.
                # Never remove an arbitrary occurrence of the word from context.
                if not word or not text.endswith(' ' + word):
                    raise ValueError(f'{prefix}: FullText must end with a space and TargetWords')
                context = text[:-(len(word) + 1)].rstrip()
                if not context:
                    raise ValueError(f'{prefix}: empty context after removing target')
                try:
                    probability = float(row[cloze_column])
                except ValueError:
                    raise ValueError(f'{prefix}: invalid {cloze_column}') from None
                if not math.isfinite(probability) or not 0 <= probability <= CLOZE_SCALES[scale]:
                    raise ValueError(f'{prefix}: {cloze_column} outside the {scale} range')
                key = (context, word)
                if key in candidates:
                    if candidates[key] != probability:
                        raise ValueError(f'{prefix}: conflicting cloze values for {key!r}')
                    continue
                candidates[key] = probability
                sentence_id = str(len(output) + 1)
                output.append([sentence_id, context, word, row[cloze_column]])
        except csv.Error as error:
            raise ValueError(f'{source}: line {reader.line_num}: invalid TSV: {error}') from error
    if not output:
        raise ValueError(f'{source}: no candidate rows')
    return output, count


def prepare_dataset(name, source_dir, output_dir, *, overwrite=False):
    relative, cloze_column, scale = PROFILES[name]
    source = Path(source_dir) / relative
    output = Path(output_dir) / f'{name}.csv'
    manifest = output.with_suffix('.csv.preparation.json')
    for path in (output, manifest):
        if path.resolve() == source.resolve():
            raise ValueError('Preparation must not overwrite source data.')
        if path.exists() and not overwrite:
            raise FileExistsError(f'{path} exists; use --overwrite to replace it.')
    source_hash = sha256(source)
    if cloze_column is None:
        # Peelle already has the prepared schema and is not participant-level data.
        load_cloze_data(source, cloze_scale=scale)
        with source.open(encoding='utf-8-sig', newline='') as file:
            rows = [[r[c] for c in REQUIRED_COLUMNS] for r in csv.DictReader(file)]
        source_rows = len(rows)
    else:
        rows, source_rows = convert_tsv(source, cloze_column, scale)
    if source_hash != sha256(source):
        raise ValueError(f'{source}: changed during preparation')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=output.parent, encoding='utf-8', newline='', delete=False) as file:
            temporary = Path(file.name)
            writer = csv.writer(file)
            writer.writerow(REQUIRED_COLUMNS)
            writer.writerows(rows)
        records = load_cloze_data(temporary, cloze_scale=scale)
        provenance = {
            'schema_version': 1, 'dataset': name, 'prepared_at': timestamp(),
            'source': {'path': str(source.resolve()), 'sha256': source_hash, 'rows': source_rows},
            'output': {'path': str(output.resolve()), 'sha256': sha256(temporary),
                       'sentences': len(records), 'candidates': len(rows), 'cloze_scale': scale},
            'removed_repeated_measurements': source_rows - len(rows),
            'context_rule': 'remove_terminal_target' if cloze_column else 'preserve_existing_context',
        }
        publish(temporary, output, overwrite=overwrite)
        write_json(manifest, provenance)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=PROFILES, default=list(PROFILES))
    parser.add_argument('--source-dir', type=Path, default=Path('data/sources'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    try:
        # Catch ordinary output collisions before starting a multi-dataset job.
        if not args.overwrite:
            for name in args.datasets:
                for suffix in ('.csv', '.csv.preparation.json'):
                    path = args.output_dir / (name + suffix)
                    if path.exists():
                        raise FileExistsError(f'{path} exists; use --overwrite or another --output-dir.')
        for name in args.datasets:
            report = prepare_dataset(name, args.source_dir, args.output_dir, overwrite=args.overwrite)
            print(f"{name}: wrote {report['output']['candidates']} candidates to {report['output']['path']}")
    except (OSError, ValueError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
