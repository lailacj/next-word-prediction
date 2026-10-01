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

| Path                                 | Contents                                                              |
| ------------------------------------ | --------------------------------------------------------------------- |
| [pipeline/](pipeline/)               | Dataset loading, model wrappers, and the pipeline entry point         |
| [data/](data/)                       | Source datasets, analysis scripts, prepared inputs, and model outputs |
| [playground/](playground/)           | Exploratory model and analysis scripts                                |
| [docs/](docs/)                       | Project document shortcut                                             |
| [requirements.txt](requirements.txt) | Python dependencies                                                   |

## Running the current pipeline

### Setup

From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install python-dotenv huggingface-hub
```

The model module imports `dotenv` and `huggingface_hub`; `python-dotenv` is currently missing from `requirements.txt`. Model weights are loaded from Hugging Face and require sufficient memory and an initial download. The Llama wrapper reads a Hugging Face token from `LLAMA_TOKEN` in the environment or a `.env` file.

### Choose an input and model

Select a prepared CSV and model from the command line:

```sh
python pipeline/run_pipeline.py --dataset data/parsed_data/szewczyk_2022.csv --model qwen
python pipeline/run_pipeline.py --dataset data/parsed_data/michaelov_2024.csv --model bert
python pipeline/run_pipeline.py --dataset data/peelle_data/cloze_data.csv --model llama
```

Supported models are `qwen`, `bert`, `deepseek`, and `llama`. Any CSV with the [prepared input schema](data/README.md#prepared-pipeline-inputs) can be supplied; no Python edits are needed.

By default, scores are written to `results/<dataset stem>/<model>.csv` under the project root. Use `--output` to choose another path:

```sh
python pipeline/run_pipeline.py --dataset data/parsed_data/nieuwland_2018.csv --model deepseek --output results/my_run.csv
```

Input and explicit output paths are relative to your working directory. Existing output files are preserved unless you pass `--overwrite`. Datasets with the same filename stem share a default output location, so use `--output` to distinguish them.

The runner reports the number of scored and skipped candidates. Use `python pipeline/run_pipeline.py --help` for options; help works without installing model dependencies. Module invocation (`python -m pipeline.run_pipeline`) also works with the same arguments.

### Runner tests

```sh
python3 -m unittest discover -s tests -v
```

These tests use a stand-in model and dataset loader to check output handling and command-line configuration without downloading weights.

### Current limitations

- Output columns named `*_prob` contain natural log probabilities.
- BERT skips words that tokenize into more than one token.
- DeepSeek currently scores each target token against the unchanged context; its multi-token scores need review before comparison with the other models.
- The SLURM launcher, [pipeline/run_pipeline.sh](pipeline/run_pipeline.sh), contains author-specific paths and cluster settings that need updating before use.
