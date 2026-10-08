# Datasets

This directory contains source data and analysis code, plus prepared cloze inputs for the [Next Word Prediction project](../README.md).

## Directory layout

| Directory | Contents |
| --- | --- |
| [sources/james-michaelov_data/](sources/james-michaelov_data/) | Preprocessed datasets and code for Michaelov and Bergen's N400 and reading-time comparison; see its [upstream README](sources/james-michaelov_data/README.md) |
| [sources/jakub_kara_data/](sources/jakub_kara_data/) | Analysis and procedure files associated with Szewczyk and Federmeier |
| [sources/james_megan_cyma_data/](sources/james_megan_cyma_data/) | Data, scripts, and analyses for *Strong Prediction* |
| [sources/peelle_data/](sources/peelle_data/) | An additional `cloze_data.csv`; its publication mapping is not specified in the supplied notes |
| [processed/](processed/) | Four prepared CSV inputs for the pipeline |
| [../results/legacy/](../results/legacy/) | Original `Model_*` output directories, preserved unchanged |

## Studies and source links

The following references and summaries come from the supplied dataset notes.

### 1. Michaelov and Bergen — N400 and reading time

**James A. Michaelov and Benjamin K. Bergen (2026).** *Better language models better model the N400, but not reading time.* Journal of Memory and Language.

- [Paper](https://www.sciencedirect.com/science/article/pii/S0749596X2600032X)
- [Code and data on OSF](https://osf.io/87gmv)
- Local files: [sources/james-michaelov_data/](sources/james-michaelov_data/)

The supplied abstract describes analyses of **four reading-time datasets and nine N400 datasets**. Larger models, models trained on more data, and models with better language-task performance better predicted N400 amplitude, while reading-time results showed a different pattern.

### 2. Szewczyk and Federmeier — Context-based semantic facilitation

**Jakub M. Szewczyk and Kara D. Federmeier (2022).** *Context-based facilitation of semantic access follows both logarithmic and linear functions of stimulus probability.* Journal of Memory and Language.

- [Paper](https://www.sciencedirect.com/science/article/pii/S0749596X21000942)
- [Code and N400 data on OSF](https://osf.io/urvax)
- Local files: [sources/jakub_kara_data/](sources/jakub_kara_data/)
- Prepared input: [processed/szewczyk_2022.csv](processed/szewczyk_2022.csv)

The study uses cloze probabilities and GPT-2 estimates of word predictability. The supplied abstract describes a reanalysis of five datasets with 138 participants, finding graded N400 facilitation even among unpredictable words. The imported Michaelov README states that `szewczyk_2022.tsv` includes all five datasets.

### 3. Michaelov and colleagues — Strong Prediction

**James A. Michaelov, Megan D. Bardolph, Cyma K. Van Petten, Benjamin K. Bergen, and Seana Coulson (2024).** *Strong Prediction: Language Model Surprisal Explains Multiple N400 Effects.* Neurobiology of Language, 5(1), 107–135.

- [Paper](https://doi.org/10.1162/nol_a_00105)
- [Code and N400 data on OSF](https://osf.io/pysbc/overview)
- Local files: [sources/james_megan_cyma_data/](sources/james_megan_cyma_data/)
- Prepared input: [processed/michaelov_2024.csv](processed/michaelov_2024.csv)

The local [N400_data.csv](sources/james_megan_cyma_data/data/N400_data.csv) has these columns:

```text
TargetWord,Condition,ContextCode,N400,Subject,PlausibilityJudgement,Electrode,Cloze,Sentence
```

**Dataset overlap:** study 1 includes data from study 3. These sources should not be counted as independent datasets without checking which observations overlap.

### 4. Nieuwland and colleagues — Large-scale replication

**Mante S. Nieuwland and colleagues (2018).** *Large-scale replication study reveals a limit on probabilistic prediction in language comprehension.* eLife, 7, e33468.

- [Paper](https://elifesciences.org/articles/33468)
- [Code and N400 data on OSF](https://osf.io/eyzaq/)
- Local source: [sources/james-michaelov_data/datasets/nieuwland_2018.tsv](sources/james-michaelov_data/datasets/nieuwland_2018.tsv)
- Prepared input: [processed/nieuwland_2018.csv](processed/nieuwland_2018.csv)

### 5. Additional N400 source — Citation to complete

The notes include another paper and dataset without an author list or title:

- [Paper DOI: 10.1037/xlm0001091](https://doi.org/10.1037/xlm0001091)
- [Code and N400 data on OSF](https://osf.io/5rtn4)

Its relationship to the local directories still needs to be established.

## Prepared pipeline inputs

The files in [processed/](processed/) share this header. `peelle.csv` is an unchanged copy of [the supplied Peelle cloze file](sources/peelle_data/cloze_data.csv):

```csv
sentence_number,sentence,word,cloze_prob
```

| Column | Meaning |
| --- | --- |
| `sentence_number` | Sentence identifier used to group candidate words |
| `sentence` | Context preceding the candidate word |
| `word` | Candidate continuation to score |
| `cloze_prob` | Human cloze value from the source dataset |

### Scale settings and normalized records

The loader uses explicit settings in `DATASET_SCALES` in [datasets.py](../src/next_word_prediction/datasets.py):

| Input filename | Input scale | Normalization |
| --- | --- | --- |
| `michaelov_2024.csv` | Proportion, 0–1 | Retain value |
| `nieuwland_2018.csv` | Percent, 0–100 | Divide by 100 |
| `szewczyk_2022.csv` | Proportion, 0–1 | Retain value |
| `peelle.csv` | Proportion, 0–1 | Retain value |

Settings match the exact filename, regardless of its directory. Unknown filenames require `--cloze-scale proportion` or `--cloze-scale percent`. An explicit scale overrides the registered setting; use it for renamed files or already-normalized copies. The loader never guesses a scale from the observed range: a percentage of `1` means `0.01`, even when all observed values are below `1`.

For Python analysis:

```python
from next_word_prediction.datasets import load_cloze_data

records = load_cloze_data("data/processed/nieuwland_2018.csv")
# For custom inputs: load_cloze_data("custom.csv", cloze_scale="percent")
for record in records:
    for candidate in record.candidates:
        print(record.sentence_id, candidate.word, candidate.cloze_prob)
```

The loader returns `SentenceRecord` objects with `sentence_id`, `sentence`, and a tuple of `Candidate` objects (`word`, `cloze_prob`). Cloze probabilities are floats in 0–1. Sentence IDs remain strings and need not be consecutive or numeric. Sentences follow first appearance in the CSV; candidates retain their order within each sentence. Duplicate candidate rows are retained.

### Validation and text handling

Before loading a model or writing scores, the loader rejects:

- Missing, blank, or duplicate column names; all four required columns must be present.
- Rows whose field counts do not match the header, and malformed CSV quoting.
- Blank required fields or a file with no candidate rows.
- Nonnumeric, nonfinite, or out-of-range cloze values for the selected scale.
- A sentence ID associated with conflicting sentence text.

Errors identify the file and physical line (the ending line for multiline records). Extra named columns are allowed and ignored, and UTF-8 files with a byte-order mark are accepted.

Text and IDs are preserved after normal CSV decoding, including literal quotes, apostrophes, and whitespace. The previous loader stripped boundary quotes and apostrophes. Preserving them changes one context in the supplied Peelle data, so scores for that context can differ on rerun even though the model scoring code is unchanged.

Source files are never rewritten. Normalization happens in memory; the score output schema remains unchanged and does not include the normalized cloze values. Use the structured records above for analysis.

## Preparing inputs from source

Use `prepare-cloze-data` (or `python -m next_word_prediction.prepare`) from the repository root:

```sh
prepare-cloze-data --output-dir data/rebuilt
```

The default source directory is `data/sources/`. The command supports all four datasets; select a subset with `--datasets` and choose another source root with `--source-dir`. Existing output CSVs and preparation manifests require `--overwrite` to replace.

For Michaelov, Nieuwland, and Szewczyk, the source TSV's `FullText` includes the target. Preparation verifies that it ends with a space followed by `TargetWords`, removes that terminal target, and deduplicates repeated participant/electrode measurements by context and target. It preserves distinct alternatives and rejects conflicting cloze values. Each distinct context/target pair receives an ID in first-seen order, matching the existing prepared files. Peelle already has the prepared schema, so its IDs and candidate rows are retained.

Preparation preserves the source cloze scale; normalization remains the loader's responsibility. All generated CSVs are validated before publication. A `.csv.preparation.json` sidecar records source/output SHA-256 hashes, candidate counts, removed repeated measurements, and the context rule. Validation against the imported sources reproduced all 58,146 candidate records exactly without rewriting the checked-in files.

The original [data_parsing.py](sources/james-michaelov_data/data_organization/data_parsing.py) remains an upstream reference; its schema and target handling are superseded by the maintained preparation command.

## Model outputs

New runs write to `results/` under the current working directory by default. Run from the repository root to keep them in this checkout:

```text
results/<dataset stem>/<model>.csv
```

For example, scoring `processed/szewczyk_2022.csv` with Qwen produces `results/szewczyk_2022/qwen.csv`. Pass `--output` to choose another location and `--overwrite` to replace an existing file. The runner owns output writing; constructing a model does not create or truncate result files.

Existing results have moved unchanged to [../results/legacy/](../results/legacy/), retaining their `Model_*` directory names and `<model>_data.csv` filenames. See the [results guide](../results/README.md).

Output headers retain the existing schema:

```csv
sentence_num,sentence,word,<model>_prob
```

Despite the `*_prob` names, the current wrappers return natural log probabilities. Surprisal in nats is the negative of that value. Candidates the model cannot score are omitted and counted in the runner's summary.
