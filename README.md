# Next Word Prediction

This project investigates how closely language model predictions align with human word prediction and language processing. It compares model log probabilities and surprisal with human cloze probabilities, N400 amplitude, reading times, reaction times, and priming.

## Central research question

**Do language model next-word probabilities become more human-like with scale, or does alignment depend on the human measure?**

Cloze tasks measure how often people supply a particular continuation. N400 amplitude, reading times, reaction times, and priming reflect related aspects of language processing. Models produce logits, which can be converted to log probabilities and surprisal.

Across different language models, we investigate the following questions:

1. Does model log probability correlate with human cloze probability?
2. Does model surprisal predict N400 amplitude?
3. Does model surprisal predict reading time, reaction time, or priming?
4. How do these relationships differ when the model is held constant?
5. Do model size, word frequency, cloze entropy, semantic similarity, and item type moderate these relationships?
6. Does cloze mediate the relationship between models and human measures, or do models explain N400 and reaction-time variance beyond cloze?

## Research background

The project notes highlight several strands of existing work (jack will read more so this will be expanded on):

- **Ryskin and Nieuwland:** prediction is adaptive and depends on memory, attention, and experience. Human measures are related but reflect different tasks, timescales, and mechanisms.
- **Michaelov and Bergen (2026):** larger and better language models better predict N400 amplitude, but do not show the same advantage for reading time.
- **Oh and Schuler:** larger transformer models can provide surprisal estimates that fit reading times worse; the notes describe a possible sweet spot around two billion training tokens.
- **“Clozing the Gap”:** model surprisal may outperform cloze because cloze has limited resolution, particularly for low-probability and semantically similar alternatives.

See the [dataset guide](data/README.md) for the paper and data links supplied with the project. Full references for the other works above remain to be added.

### Working hypotheses

> We can have some hypotheses here but since this is more investigative, we may have to read more first and then can develop more grounded hypotheses

<!-- Better next-word prediction may not consistently imply more human-like language processing. Scaling may improve alignment with N400 and semantic preactivation while reducing alignment with reading-time behavior, particularly for rare words, named entities, low-cloze continuations, and high-entropy contexts. This is a hypothesis to test, not a result established by this repository. -->

## Project structure

The work has two main parts:

1. **Literature review:** reading times, cloze probability, reaction times, priming, surprisal theory, N400, and model logits.
2. **Computational pipeline:** score candidate words in sentence contexts with language models for comparison with human measures.

```text
src/next_word_prediction/   Reusable package: models, datasets, pipeline, CLI
scripts/                   Cluster launcher and workflow scripts
data/sources/              Imported datasets and upstream code
data/processed/            Prepared pipeline inputs
results/legacy/            Existing scores preserved from data/Model_*
tests/                     Automated correctness tests
playground/                New experiments using the shared package
archive/                   Original standalone runners and one-off scripts
docs/                      Project document and original notes
pyproject.toml             Package metadata, dependencies, and CLI entry point
```

See the [dataset guide](data/README.md), [results guide](results/README.md), [playground guide](playground/README.md), and [script instructions](scripts/README.md).

## Running the current pipeline

### Setup

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Dependencies are defined in [pyproject.toml](pyproject.toml). For plotting and exploratory analysis, use `python -m pip install -e ".[analysis]"`. The compatibility `requirements.txt` installs the package with those extras. Model weights are loaded from Hugging Face and require sufficient memory and an initial download. The Llama wrapper reads a Hugging Face token from `LLAMA_TOKEN` in the environment or a `.env` file.

### Choose an input and model

Select a prepared CSV and model from the command line:

```sh
next-word-prediction --dataset data/processed/szewczyk_2022.csv --model qwen
next-word-prediction --dataset data/processed/michaelov_2024.csv --model bert
next-word-prediction --dataset data/processed/peelle.csv --model llama
```

Supported models are `qwen`, `bert`, `deepseek`, and `llama`. Any CSV with the [prepared input schema](data/README.md#prepared-pipeline-inputs) can be supplied; no Python edits are needed.

The four prepared filenames have explicit cloze-scale settings: Nieuwland uses percentages, and Michaelov, Szewczyk, and Peelle use proportions. For another filename, specify its input scale:

```sh
next-word-prediction --dataset my_cloze.csv --cloze-scale percent --model bert
```

`--cloze-scale proportion` expects 0–1; `--cloze-scale percent` expects 0–100. An explicit flag overrides the registered filename setting, including for an already-normalized copy. Scale is never inferred from observed values. All rows are validated before model loading or output writing.


By default, scores are written to `results/<dataset stem>/<model>.csv` under the current working directory. Use `--output` to choose another path:

```sh
next-word-prediction --dataset data/processed/nieuwland_2018.csv --model deepseek --output results/my_run.csv
```

Input and explicit output paths are relative to your working directory. Existing output files are preserved unless you pass `--overwrite`. Datasets with the same filename stem share a default output location, so use `--output` to distinguish them.

The runner reports the number of scored and skipped candidates. Use `next-word-prediction --help` for options; help works without installing model dependencies. Module invocation (`python -m next_word_prediction`) also works with the same arguments.

### Model interface

[src/next_word_prediction/models.py](src/next_word_prediction/models.py) provides one `LanguageModel` class for all four models:

```python
from next_word_prediction import LanguageModel

model = LanguageModel("qwen")
context = model.tokenize_sentence("The capital of France is")
word_tokens = model.tokenize_word("Paris")
if word_tokens:
    log_probability = model.predict_next_word(context, word_tokens)
```

`MODEL_IDS` defines the supported names and Hugging Face checkpoints and supplies the CLI's model choices. Loading and word tokenization are shared; model-specific cases handle sentence preparation, masked scoring, and causal scoring. Constructing a model loads weights and explicitly sets evaluation mode, disabling training-time dropout. It does not write results.

Model checkpoints and prompt tokenization are unchanged. DeepSeek now conditions each continuation token on the preceding tokens; see the scoring correction below.

### Tests

After installing the package, run both suites:

```sh
python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest discover -s tests/integration -v
```

The integration suite is run separately and requires PyTorch and Transformers. It creates tiny randomly initialized Qwen2 and Llama models locally; no pretrained weights or tokenizers are downloaded. Missing dependencies cause this suite to fail rather than silently skip validation.

For the lightweight suite and CLI help alone, you can skip ML dependency installation:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m next_word_prediction --help
```

The lightweight tests check dataset validation, runner output handling, and model dispatch using fixtures and stand-ins.

The numerical integration tests exercise the production Qwen, DeepSeek, and Llama scoring methods with real CPU tensors and tiny models. They compare single-token and multi-token scores against an independent, uncached forward pass over the entire prompt and continuation, with an absolute tolerance of `1e-5` in summed natural log probability. They also check DeepSeek attention masks, Llama's cache, candidate-order independence, unchanged context tensors, and evaluation mode.

Validation passed with Python 3.14, PyTorch 2.14.1, and Transformers 5.18.0: 30 lightweight tests and four integration tests. This validates causal scoring for supplied token IDs; it does not validate pretrained tokenizers, sentence–word boundaries, full-size checkpoints, GPU execution, or BERT numerical scoring.

### DeepSeek scoring correction

DeepSeek now computes multi-token word scores as the sum of conditional log probabilities:

```text
log P(t1 | sentence) + log P(t2 | sentence, t1) + ...
```

The input IDs and attention mask grow between token predictions. Each candidate uses a local context mapping, so scoring it does not change the context shared with other candidates. Single-token scoring follows the same computation as before.

Earlier versions incorrectly scored every token against the original sentence. Existing multi-token DeepSeek scores need regeneration before comparison or analysis. Existing result files have not been recomputed or modified.

Regression tests cover context-dependent scores, attention-mask growth, candidate isolation, and single-token behavior. The integration suite also compares DeepSeek scores with an independent full-sequence calculation using a tiny random Qwen2 model.

### Current limitations

- Output columns named `*_prob` contain natural log probabilities.
- BERT skips words that tokenize into more than one token.
- The [SLURM launcher](scripts/run_pipeline.sh) uses cluster-specific resource settings; see [scripts/README.md](scripts/README.md) before submitting a job.

## Migration from the previous layout

- Replace `python pipeline/run_pipeline.py` with `next-word-prediction` after installing the package.
- Replace `from pipeline.language_models import LanguageModel` with `from next_word_prediction import LanguageModel`.
- Replace `data/parsed_data/` paths with `data/processed/`; Peelle's prepared input is `data/processed/peelle.csv`.
- Imported source collections are under `data/sources/`; existing `Model_*` outputs are under `results/legacy/`.
- Default output paths are relative to the working directory, so installed commands work outside this checkout. Run from the repository root to keep results here.
- Original temporary README notes are preserved in [docs/notes/](docs/notes/).
