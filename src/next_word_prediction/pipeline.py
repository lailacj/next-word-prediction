"""Load cloze inputs, score candidates, and write results."""

import csv
from pathlib import Path

from .models import LanguageModel


def create_model(name):
    return LanguageModel(name)


def load_dataset(path, *, cloze_scale=None):
    from .datasets import load_cloze_data

    return load_cloze_data(path, cloze_scale=cloze_scale)


def run_pipeline(dataset, model_name, output, *, overwrite=False, cloze_scale=None):
    """Write scores and return the number of scored and skipped candidates."""
    dataset = Path(dataset)
    output = Path(output)
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset}")
    if dataset.resolve() == output.resolve():
        raise ValueError("The output path must differ from the dataset path.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")

    records = load_dataset(dataset, cloze_scale=cloze_scale)
    model = create_model(model_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored = skipped = 0

    # Exclusive creation also prevents overwriting a file created during model loading.
    with output.open("w" if overwrite else "x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sentence_num", "sentence", "word", f"{model_name}_prob"])
        for record in records:
            sentence_token_ids = model.tokenize_sentence(record.sentence)
            for candidate in record.candidates:
                word_token_ids = model.tokenize_word(candidate.word)
                if not word_token_ids:
                    skipped += 1
                    continue
                score = model.predict_next_word(sentence_token_ids, word_token_ids)
                if score is None:
                    skipped += 1
                    continue
                writer.writerow([record.sentence_id, record.sentence, candidate.word, score])
                scored += 1

    return scored, skipped
