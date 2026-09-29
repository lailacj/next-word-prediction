# Datasets

This directory contains source data and analysis code, prepared cloze inputs, and language model outputs for the [Next Word Prediction project](../README.md).

## Directory layout

| Directory | Contents |
| --- | --- |
| [james-michaelov_data/](james-michaelov_data/) | Preprocessed datasets and code for Michaelov and Bergen's N400 and reading-time comparison; see its [upstream README](james-michaelov_data/README.md) |
| [jakub_kara_data/](jakub_kara_data/) | Analysis and procedure files associated with Szewczyk and Federmeier |
| [james_megan_cyma_data/](james_megan_cyma_data/) | Data, scripts, and analyses for *Strong Prediction* |
| [peelle_data/](peelle_data/) | An additional `cloze_data.csv`; its publication mapping is not specified in the supplied notes |
| [parsed_data/](parsed_data/) | Three prepared CSV inputs for the pipeline |
| `Model_michaelov/`, `Model_nieuwland/`, `Model_szewczyk/`, `Model_peelle/` | Model outputs, with subdirectories for individual models |

## Studies and source links

The following references and summaries come from the supplied dataset notes.

### 1. Michaelov and Bergen — N400 and reading time

**James A. Michaelov and Benjamin K. Bergen (2026).** *Better language models better model the N400, but not reading time.* Journal of Memory and Language.

- [Paper](https://www.sciencedirect.com/science/article/pii/S0749596X2600032X)
- [Code and data on OSF](https://osf.io/87gmv)
- Local files: [james-michaelov_data/](james-michaelov_data/)

The supplied abstract describes analyses of **four reading-time datasets and nine N400 datasets**. Larger models, models trained on more data, and models with better language-task performance better predicted N400 amplitude, while reading-time results showed a different pattern.

### 2. Szewczyk and Federmeier — Context-based semantic facilitation

**Jakub M. Szewczyk and Kara D. Federmeier (2022).** *Context-based facilitation of semantic access follows both logarithmic and linear functions of stimulus probability.* Journal of Memory and Language.

- [Paper](https://www.sciencedirect.com/science/article/pii/S0749596X21000942)
- [Code and N400 data on OSF](https://osf.io/urvax)
- Local files: [jakub_kara_data/](jakub_kara_data/)
- Prepared input: [parsed_data/szewczyk_2022.csv](parsed_data/szewczyk_2022.csv)

The study uses cloze probabilities and GPT-2 estimates of word predictability. The supplied abstract describes a reanalysis of five datasets with 138 participants, finding graded N400 facilitation even among unpredictable words. The imported Michaelov README states that `szewczyk_2022.tsv` includes all five datasets.

### 3. Michaelov and colleagues — Strong Prediction

**James A. Michaelov, Megan D. Bardolph, Cyma K. Van Petten, Benjamin K. Bergen, and Seana Coulson (2024).** *Strong Prediction: Language Model Surprisal Explains Multiple N400 Effects.* Neurobiology of Language, 5(1), 107–135.

- [Paper](https://doi.org/10.1162/nol_a_00105)
- [Code and N400 data on OSF](https://osf.io/pysbc/overview)
- Local files: [james_megan_cyma_data/](james_megan_cyma_data/)
- Prepared input: [parsed_data/michaelov_2024.csv](parsed_data/michaelov_2024.csv)

The local [N400_data.csv](james_megan_cyma_data/data/N400_data.csv) has these columns:

```text
TargetWord,Condition,ContextCode,N400,Subject,PlausibilityJudgement,Electrode,Cloze,Sentence
```

**Dataset overlap:** study 1 includes data from study 3. These sources should not be counted as independent datasets without checking which observations overlap.

### 4. Nieuwland and colleagues — Large-scale replication

**Mante S. Nieuwland and colleagues (2018).** *Large-scale replication study reveals a limit on probabilistic prediction in language comprehension.* eLife, 7, e33468.

- [Paper](https://elifesciences.org/articles/33468)
- [Code and N400 data on OSF](https://osf.io/eyzaq/)
- Local source: [james-michaelov_data/datasets/nieuwland_2018.tsv](james-michaelov_data/datasets/nieuwland_2018.tsv)
- Prepared input: [parsed_data/nieuwland_2018.csv](parsed_data/nieuwland_2018.csv)

### 5. Additional N400 source — Citation to complete

The notes include another paper and dataset without an author list or title:

- [Paper DOI: 10.1037/xlm0001091](https://doi.org/10.1037/xlm0001091)
- [Code and N400 data on OSF](https://osf.io/5rtn4)

Its relationship to the local directories still needs to be established.

## Prepared pipeline inputs

The files in [parsed_data/](parsed_data/) share this header:

```csv
sentence_number,sentence,word,cloze_prob
```

| Column | Meaning |
| --- | --- |
| `sentence_number` | Sentence identifier used to group candidate words |
| `sentence` | Context preceding the candidate word |
| `word` | Candidate continuation to score |
| `cloze_prob` | Human cloze value from the source dataset |

The loader groups words by sentence identifier. The current runner assumes identifiers are consecutive integers beginning at 1, encountered in that order.

**Cloze values need normalization before cross-dataset analysis.** For example, the Nieuwland input contains values such as `100` and `90`, while the Michaelov and Szewczyk inputs contain fractional values. The loader retains these values as strings and does not normalize them.

The source [data_parsing.py](james-michaelov_data/data_organization/data_parsing.py) writes to `james-michaelov_data/parsed_data/` with a different header (`sentence_num,FullText,target_word,cloz`). Its output is not directly compatible with the current pipeline loader.

## Model outputs

Outputs are organized as:

```text
Model_<dataset>/<model>/<model>_data.csv
```

The current Qwen wrapper instead writes `qwen_datas.csv`, while the checked-in Qwen files are named `qwen_data.csv`.

Output headers follow this pattern:

```csv
sentence_num,sentence,word,<model>_prob
```

Despite the `*_prob` names, the current wrappers return natural log probabilities. Surprisal in nats is the negative of that value. Model initialization overwrites its output file; select the appropriate path before running a new dataset/model combination.
