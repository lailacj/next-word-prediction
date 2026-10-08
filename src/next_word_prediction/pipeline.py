"""Validate inputs, score candidates, and publish results with run metadata."""

import csv
import math
from pathlib import Path
import tempfile
import uuid

from .artifacts import publish, sha256, write_json
from .datasets import DATASET_SCALES, load_cloze_data
from .models import LanguageModel
from .provenance import code_version, environment, timestamp


def create_model(name, **settings):
    return LanguageModel(name, **settings)


def load_dataset(path, *, cloze_scale=None):
    return load_cloze_data(path, cloze_scale=cloze_scale)


def run_pipeline(dataset, model_name, output, *, overwrite=False, cloze_scale=None,
                 device="cpu", dtype="float32", revision="main"):
    """Publish a complete score CSV; metadata distinguishes failed runs."""
    dataset, output = Path(dataset), Path(output)
    metadata_path = output.with_suffix(output.suffix + '.metadata.json')
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset}")
    if dataset.resolve() in (output.resolve(), metadata_path.resolve()):
        raise ValueError("Output and metadata paths must differ from the dataset path.")
    for path in (output, metadata_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}. Use --overwrite to replace it.")

    input_hash = sha256(dataset)
    records = load_dataset(dataset, cloze_scale=cloze_scale)
    if sha256(dataset) != input_hash:
        raise ValueError("Dataset changed during loading; retry with a stable input file.")
    total = sum(len(record.candidates) for record in records)
    metadata = {
        "schema_version": 1, "run_id": str(uuid.uuid4()), "status": "loading_model",
        "started_at": timestamp(), "finished_at": None,
        "input": {"path": str(dataset.resolve()), "sha256": input_hash,
                  "cloze_scale": cloze_scale or DATASET_SCALES.get(dataset.name),
                  "sentences": len(records), "candidates": total},
        "output": {"path": str(output.resolve()), "sha256": None, "score_units": "natural_log_probability"},
        "settings": {"model": model_name, "device": device, "dtype": dtype, "revision": revision},
        "model": None, "environment": environment(), "code": code_version(),
        "counts": {"scored": 0, "skipped": 0, "skip_reasons": {"empty_tokens": 0, "unsupported_word": 0}},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the sidecar to prevent two runs from publishing to the same path.
    with metadata_path.open('w' if overwrite else 'x', encoding='utf-8'):
        pass
    temporary = None
    counts = metadata['counts']
    try:
        write_json(metadata_path, metadata)
        model = create_model(model_name, device=device, dtype=dtype, revision=revision)
        metadata['model'] = model.run_metadata()
        metadata['status'] = 'running'
        write_json(metadata_path, metadata)
        with tempfile.NamedTemporaryFile('w', dir=output.parent, encoding='utf-8', newline='', delete=False) as file:
            temporary = Path(file.name)
            writer = csv.writer(file)
            writer.writerow(['sentence_num', 'sentence', 'word', f'{model_name}_prob'])
            for record in records:
                context = model.tokenize_sentence(record.sentence)
                for candidate in record.candidates:
                    tokens = model.tokenize_word(candidate.word)
                    if not tokens:
                        counts['skipped'] += 1
                        counts['skip_reasons']['empty_tokens'] += 1
                        continue
                    score = model.predict_next_word(context, tokens)
                    if score is None:
                        counts['skipped'] += 1
                        counts['skip_reasons']['unsupported_word'] += 1
                        continue
                    if not math.isfinite(score):
                        raise ValueError(f"Nonfinite score for sentence {record.sentence_id!r}, word {candidate.word!r}")
                    writer.writerow([record.sentence_id, record.sentence, candidate.word, score])
                    counts['scored'] += 1
        result_hash = sha256(temporary)
        publish(temporary, output, overwrite=overwrite)
        metadata['output']['sha256'] = result_hash
        metadata['status'] = 'complete'
    except BaseException as error:
        metadata['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
        metadata['error_type'] = type(error).__name__
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        metadata['finished_at'] = timestamp()
        write_json(metadata_path, metadata)
    return counts['scored'], counts['skipped']
