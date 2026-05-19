import csv
from pathlib import Path
from typing import Iterable, Optional


# sanitize the close_prof field accross all the datasets
def _pick_cloze_field(fieldnames: Iterable[str]) -> Optional[str]:
	normalized = {name.strip().lower() for name in fieldnames if name}
	if "cloze" in normalized:
		return next(name for name in fieldnames if name.strip().lower() == "cloze")
	if "cloze_p" in normalized:
		return next(name for name in fieldnames if name.strip().lower() == "cloze_p")
	if "cloz" in normalized:
		return next(name for name in fieldnames if name.strip().lower() == "cloz")
	return None


def data_parsing() -> None:
	"""Parse all dataset TSVs into clean CSVs with a unified schema."""
	base_dir = Path(__file__).resolve().parents[1]
	datasets_dir = base_dir / "datasets"
	parsed_dir = base_dir / "parsed_data"
	parsed_dir.mkdir(parents=True, exist_ok=True)

	allowed_files = {
		"michaelov_2024.tsv",
		"nieuwland_2018.tsv",
		"szewczyk_2022.tsv",
	}

	for tsv_path in sorted(datasets_dir.glob("*.tsv")):
		if tsv_path.name not in allowed_files:
			continue

		with tsv_path.open("r", encoding="utf-8", newline="") as tsv_file:
			reader = csv.DictReader(tsv_file, delimiter="\t")
			if not reader.fieldnames:
				continue

			required_fields = {"FullText", "TargetWords"}
			missing_fields = required_fields - set(reader.fieldnames)
			if missing_fields:
				raise ValueError(
					f"{tsv_path.name} is missing required fields: {sorted(missing_fields)}"
				)

			cloze_field = _pick_cloze_field(reader.fieldnames)
			if cloze_field is None:
				raise ValueError(
					f"{tsv_path.name} is missing a cloze field (cloze, cloze_p, cloz)."
				)

			output_path = parsed_dir / f"{tsv_path.stem}.csv"
			with output_path.open("w", encoding="utf-8", newline="") as csv_file:
				writer = csv.DictWriter(
					csv_file,
					fieldnames=["sentence_num", "FullText", "target_word", "cloz"],
				)
				writer.writeheader()

				seen_sentences = set()
				idx = 1
				for row in reader:
					full_text = row.get("FullText")
					target_word = row.get("TargetWords")
					cloze_value = row.get(cloze_field)
					if full_text is None or target_word is None or cloze_value is None:
						continue
					
					full_text_clean = full_text.strip()
					if full_text_clean in seen_sentences:
						continue
					seen_sentences.add(full_text_clean)

					writer.writerow(
						{
							"sentence_num": idx,
							"FullText": full_text_clean,
							"target_word": target_word.strip(),
							"cloz": cloze_value,
						}
					)
					idx += 1


if __name__ == "__main__":
	data_parsing()
