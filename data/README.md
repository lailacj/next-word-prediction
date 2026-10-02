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

The loader groups words by sentence identifier. The runner preserves these identifiers in the output; they do not need to be consecutive.

**Cloze values need normalization before cross-dataset analysis.** For example, the Nieuwland input contains values such as `100` and `90`, while the Michaelov and Szewczyk inputs contain fractional values. The loader retains these values as strings and does not normalize them.

The source [data_parsing.py](sources/james-michaelov_data/data_organization/data_parsing.py) writes to `sources/james-michaelov_data/parsed_data/` with a different header (`sentence_num,FullText,target_word,cloz`). Its output is not directly compatible with the current pipeline loader.

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
