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

Configuration currently requires editing Python files:

1. In [pipeline/run_pipeline.py](pipeline/run_pipeline.py), set `file_path` to a prepared dataset and choose `QwenModel`, `BertModel`, `DeepSeekModel`, or `LlamaModel`. The current defaults are `szewczyk_2022.csv` and `QwenModel()`.
2. In [pipeline/language_models.py](pipeline/language_models.py), set the selected model's `self.output_file` to the matching dataset directory. All four wrappers currently target `data/Model_szewczyk/`.
3. Run one dataset/model combination at a time:

   ```sh
   python pipeline/run_pipeline.py
   ```

Prepared inputs and output locations are documented in the [dataset guide](data/README.md). Model initialization overwrites the selected output file with a new header, so preserve any results you need before rerunning.

### Current limitations

- Output columns named `*_prob` contain natural log probabilities.
- BERT skips words that tokenize into more than one token.
- DeepSeek currently scores each target token against the unchanged context; its multi-token scores need review before comparison with the other models.
- The SLURM launcher, [pipeline/run_pipeline.sh](pipeline/run_pipeline.sh), contains author-specific paths and cluster settings that need updating before use.
