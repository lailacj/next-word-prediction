"""Validate prepared cloze CSVs and normalize explicitly configured scales."""

import csv
from dataclasses import dataclass
import math
from pathlib import Path

CLOZE_SCALES = {"proportion": 1.0, "percent": 100.0}
DATASET_SCALES = {
    "michaelov_2024.csv": "proportion",
    "nieuwland_2018.csv": "percent",
    "szewczyk_2022.csv": "proportion",
    "peelle.csv": "proportion",
}
REQUIRED_COLUMNS = ("sentence_number", "sentence", "word", "cloze_prob")


@dataclass(frozen=True)
class Candidate:
    word: str
    cloze_prob: float


@dataclass(frozen=True)
class SentenceRecord:
    sentence_id: str
    sentence: str
    candidates: tuple[Candidate, ...]


def load_cloze_data(file_path: str | Path, *, cloze_scale: str | None = None) -> list[SentenceRecord]:
    """Read all rows before returning records in first-seen sentence order.

    Known filenames have explicit scale settings. Other inputs must supply
    cloze_scale='proportion' or 'percent'. Text and IDs are preserved exactly;
    CSV quoting is decoded by the CSV parser. Extra named columns are allowed.
    """
    path = Path(file_path)
    scale = cloze_scale if cloze_scale is not None else DATASET_SCALES.get(path.name)
    if scale not in CLOZE_SCALES:
        raise ValueError(
            f"{path}: choose a cloze scale with --cloze-scale proportion or percent "
            "(or pass cloze_scale to load_cloze_data)."
        )
    maximum = CLOZE_SCALES[scale]
    contexts: dict[str, str] = {}
    candidates: dict[str, list[Candidate]] = {}

    def fail(line, message):
        return ValueError(f"{path}: line {line}: {message}")

    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, strict=True)
        try:
            header = reader.fieldnames
            if not header:
                raise fail(1, "missing CSV header")
            if len(header) != len(set(header)) or any(not name.strip() for name in header):
                raise fail(1, "column names must be nonempty and unique")
            missing = set(REQUIRED_COLUMNS) - set(header)
            if missing:
                raise fail(1, f"missing required columns: {', '.join(sorted(missing))}")

            for row in reader:
                line = reader.line_num
                if None in row or any(value is None for value in row.values()):
                    raise fail(line, "row has a different number of fields than the header")
                for column in REQUIRED_COLUMNS:
                    if not row[column].strip():
                        raise fail(line, f"{column} must not be blank")
                sentence_id = row["sentence_number"]
                sentence = row["sentence"]
                if sentence_id in contexts and contexts[sentence_id] != sentence:
                    raise fail(line, f"sentence_number {sentence_id!r} has conflicting sentence text")
                try:
                    value = float(row["cloze_prob"])
                except ValueError:
                    raise fail(line, f"cloze_prob must be numeric, got {row['cloze_prob']!r}") from None
                if not math.isfinite(value) or not 0 <= value <= maximum:
                    raise fail(line, f"cloze_prob must be finite and between 0 and {maximum:g} for {scale}")
                contexts[sentence_id] = sentence
                candidates.setdefault(sentence_id, []).append(Candidate(row["word"], value / maximum))
        except csv.Error as error:
            raise fail(reader.line_num, f"invalid CSV: {error}") from error

    if not contexts:
        raise fail(reader.line_num, "dataset contains no candidate rows")
    return [SentenceRecord(sid, sentence, tuple(candidates[sid])) for sid, sentence in contexts.items()]
