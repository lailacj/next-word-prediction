# Scripts

## SLURM launcher

Install the package in an environment available on the compute node, activate that environment, and submit from the repository root:

```sh
sbatch scripts/run_pipeline.sh --dataset data/processed/szewczyk_2022.csv --model qwen --device cuda --dtype bfloat16
```

The launcher forwards arguments to `python -m next_word_prediction`. Input paths and the default `results/` directory are relative to the submission directory. Logs are written there as `next-word-prediction.<job ID>.out` and `.err`.

Adjust the partition, resource requests, and time limit for your cluster. The launcher requests one GPU. Pass `--device cuda` to use it and choose a supported `--dtype`; allocation alone does not select the device.

For data preparation, use `prepare-cloze-data`; see the [dataset guide](../data/README.md#preparing-inputs-from-source). Imported upstream scripts remain preserved as references.

## Tokenizer audit

After installing the package, audit token boundaries without loading model weights:

```sh
python scripts/audit_tokenization.py --models qwen deepseek bert --datasets data/processed/*.csv
```

The first run downloads tokenizer files. Add `--local-files-only` for an offline run with cached tokenizers. `HF_HOME` can point to a chosen cache directory. Add `llama` to `--models` when your Hugging Face account has access to its gated repository; omitting `--models` requests all four models. Use `--revision <commit>` with one model to reproduce an exact tokenizer revision recorded in the report.

The script prints JSON containing input hashes, resolved tokenizer revisions, candidate token counts, and separate-versus-joint tokenization mismatches. It uses the production token preparation methods. BERT reports token coverage instead of a causal boundary comparison. Exit status is nonzero if a requested tokenizer is unavailable or an input produces a boundary mismatch, an empty encoding, or an unknown token. Expected multi-token BERT exclusions do not make the audit fail.

The reviewed [audit report](../docs/tokenization_audit.json) includes the unavailable Llama tokenizer explicitly. It covers Qwen, DeepSeek, and BERT across all four prepared datasets.

## Complete scoring workflow

```sh
bash scripts/run_all.sh --device cuda --dtype float32
```

Run from the repository root with the package environment activated. The script checks that all four prepared CSVs exist, then runs each supported model on each dataset in a separate process (16 runs). It stops on the first failure, including inaccessible checkpoints or existing output files. Configure Llama access beforehand. `--device`, `--dtype`, and `--overwrite` are forwarded to each run; use the individual CLI to select a model, dataset, output, or pinned revision. Do not run multiple instances against the same output directory with `--overwrite`.
