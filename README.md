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

Dependencies are defined in [pyproject.toml](pyproject.toml). For plotting and exploratory analysis, use `python -m pip install -e ".[analysis]"`. The compatibility `requirements.txt` installs the package with those extras. The [validation constraints](constraints-validation.txt) record the exact package versions used for CPU checks; use `python -m pip install -e . -c constraints-validation.txt` to request that version set. Wheel availability and accelerator support depend on the platform. Model weights are loaded from Hugging Face and require sufficient memory and an initial download. For Llama, configure an authorized Hugging Face login or `HF_TOKEN`; the existing `LLAMA_TOKEN` alias (including `.env`) is also supported. Loading does not change your global login or prompt interactively.

### Prepare inputs

The four validated CSVs in `data/processed/` are ready to use. To rebuild them from the imported sources, use the maintained preparation command:

```sh
prepare-cloze-data --output-dir data/rebuilt
```

This writes CSVs and `.csv.preparation.json` files containing source/output hashes, source row counts, cloze scales, and transformation details. It reproduces the existing structured records exactly. Use `--datasets michaelov_2024` for one dataset or `--overwrite` to intentionally replace outputs. See the [dataset guide](data/README.md) for the conversion rules. The original source files and upstream scripts remain unchanged.

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

Input and explicit output paths are relative to your working directory. Existing score CSVs and metadata files are preserved unless you pass `--overwrite`. Datasets with the same filename stem share a default output location, so use `--output` to distinguish them.

Each run writes a `.csv.metadata.json` sidecar with input and output hashes, model/tokenizer revisions, device and precision, package versions, source-code hashes, Git state, timestamps, and scored/skipped counts. Scores are staged and published only when inference completes successfully. Failed or interrupted runs leave a sidecar with that status; use only outputs with matching hashes and `status: complete`. See the [results guide](results/README.md).

The runner reports the number of scored and skipped candidates. Use `next-word-prediction --help` for options; help works without installing model dependencies. Module invocation (`python -m next_word_prediction`) also works with the same arguments.

### Device, precision, and repeat runs

Defaults are explicit: CPU, `float32`, and revision `main`. Choose an accelerator and precision for your machine:

```sh
next-word-prediction --dataset data/processed/michaelov_2024.csv --model bert --device cuda --dtype float32
next-word-prediction --dataset data/processed/szewczyk_2022.csv --model qwen --device cuda:0 --dtype bfloat16
```

`--device` accepts `cpu`, `cuda`, `cuda:N`, and `mps`; `--dtype` accepts `float32`, `float16`, and `bfloat16`. Requested accelerators must be available. Precision support and memory capacity still depend on the hardware. The runner uses one device per process; it does not shard models across GPUs.

Use `--revision <commit>` to repeat the checkpoint recorded in a previous run. The tokenizer loads from the same resolved model commit. Package versions and source hashes are recorded, but exact numerical reproduction across different hardware is not guaranteed.

After configuring the environment and Llama access, the complete 16-run workflow is:

```sh
bash scripts/run_all.sh --device cuda --dtype float32
```

It runs four datasets for each of BERT, DeepSeek, Qwen, and Llama sequentially and stops on the first failure. It preserves existing outputs by default. Choose device and precision for the intended machine before starting; the cleanup itself does not launch this workflow.

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

Model checkpoints are unchanged. Qwen now uses one separating space between the prompt and candidate, and DeepSeek conditions each continuation token on the preceding tokens; see the corrections below.

### Tests

After installing the package, run both suites:

```sh
python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest discover -s tests/integration -v
```

The integration suite is run separately and requires PyTorch and Transformers. It creates tiny randomly initialized Qwen2, Llama, and BERT models plus a small local WordPiece tokenizer; no pretrained weights or tokenizers are downloaded. Missing dependencies cause this suite to fail rather than silently skip validation.

For the lightweight suite and CLI help alone, you can skip ML dependency installation:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m next_word_prediction --help
```

The lightweight tests check dataset validation, runner output handling, and model dispatch using fixtures and stand-ins.

The numerical integration tests exercise the production Qwen, DeepSeek, and Llama scoring methods with real CPU tensors and tiny models. They compare single-token and multi-token scores against an independent, uncached forward pass over the entire prompt and continuation, with an absolute tolerance of `1e-5` in summed natural log probability. They also check DeepSeek attention masks, Llama's cache, candidate-order independence, unchanged context tensors, and evaluation mode. A separate BERT test compares the masked-word score with a direct calculation at an independently specified mask position and exercises the full CSV-to-score-and-metadata path with that tiny model.

Validation passed with Python 3.14, PyTorch 2.14.1, and Transformers 5.19.0: 47 lightweight tests and five integration tests. These validate scoring on tiny models; full-size checkpoints and GPU execution have not been tested. The separate pretrained-tokenizer audit below checks the supplied dataset boundaries.

### Tokenization and inference settings

For causal models, the prompt is `sentence.strip()` and the continuation is `" " + word.strip()`: one separating space, with the tokenizer's configured special tokens on the prompt only. Candidates have no added special tokens. The scorer uses raw text, without a chat template. Empty or whitespace-only candidates return no token IDs.

Previously, Qwen also appended a space to the prompt, so its prompt and candidate tokens decoded to text with two separating spaces. **Regenerate Qwen scores produced with the previous preparation.** The change affects all 58,146 candidates in the supplied inputs. Existing result files have not been modified.

An audit using the actual pretrained tokenizers found:

| Model | Boundary audit across 58,146 candidates | Candidate coverage |
| --- | --- | --- |
| Qwen | Zero mismatches between separate and joint tokenization after the fix | 48,175 single-token; 9,971 multi-token |
| DeepSeek | Zero mismatches between separate and joint tokenization | 48,175 single-token; 9,971 multi-token |
| BERT | Uses masked scoring, so the causal boundary comparison does not apply | 48,078 single-token; 10,068 multi-token candidates skipped; zero unknown tokens |
| Llama | Pretrained tokenizer unavailable through unauthenticated access to the gated repository | Not measured |

The [audit report](docs/tokenization_audit.json) records the resolved tokenizer revisions, input file hashes, and per-dataset counts. See [scripts/README.md](scripts/README.md) to reproduce it. These results apply to the supplied inputs; they do not prove that independent prompt/word tokenization matches joint tokenization for arbitrary text.

BERT retains its existing `sentence + " [MASK]."` prompt, including the period to the right of the mask. It scores a single masked token with that right context; its scores should be distinguished from causal next-token scores. The period and multi-token exclusion are unchanged research choices.

Inference uses evaluation mode and `no_grad()`. All tokenized inputs move to the model's device. Qwen and DeepSeek disable cache creation because they pass the full growing context each time; Llama reuses its cache. Device and dtype are explicit CLI settings; defaults are CPU and float32. GPU allocation alone does not change them. Tokenization does not request padding or truncation; long inputs still need to fit the model's context window.

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
