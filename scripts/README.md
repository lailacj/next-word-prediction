# Scripts

## SLURM launcher

Install the package in an environment available on the compute node, activate that environment, and submit from the repository root:

```sh
sbatch scripts/run_pipeline.sh --dataset data/processed/szewczyk_2022.csv --model qwen
```

The launcher forwards arguments to `python -m next_word_prediction`. Input paths and the default `results/` directory are relative to the submission directory. Logs are written there as `next-word-prediction.<job ID>.out` and `.err`.

Adjust the partition, resource requests, and time limit for your cluster. GPU allocation alone does not make the current model wrappers use the GPU.

Source-specific preparation scripts remain with the imported collections in [data/sources/](../data/sources/); their schemas have not been standardized yet.
