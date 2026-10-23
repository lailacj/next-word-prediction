# Results

New runs default to `results/<dataset stem>/<model>.csv` relative to the working directory. Run from the repository root to save them here, or select a destination with `--output`.

[legacy/](legacy/) preserves the original `data/Model_*` directories and their CSVs unchanged. Their model-specific headers and scoring conventions may differ from future runs; directory relocation does not validate or recompute scores.

New result files are ignored by Git by default. To preserve a selected result in version control, add it explicitly with `git add -f <path>` and record how it was produced.

## Completed production batch

The October 20, 2026 Oscar batch completed all 16 model/dataset combinations
with pinned checkpoints and CUDA float32 inference. The outputs are available
both on Oscar and locally in the dataset directories below:

| Dataset | BERT | DeepSeek | Qwen | Llama |
| --- | ---: | ---: | ---: | ---: |
| [Michaelov](michaelov_2024/) | 492 | 500 | 500 | 500 |
| [Nieuwland](nieuwland_2018/) | 146 | 160 | 160 | 160 |
| [Szewczyk](szewczyk_2022/) | 1,277 | 1,330 | 1,330 | 1,330 |
| [Peelle](peelle/) | 46,163 | 56,156 | 56,156 | 56,156 |

These are score counts: 222,516 total, with 10,068 expected BERT exclusions.
The cluster's final validation and independent local verification both passed.
Each dataset directory contains four CSVs and four metadata sidecars; sidecar
paths describe the original execution on Oscar.

The [Oscar runbook](../docs/oscar.md#production-batch-october-20-2026) records
checkpoint pins, job IDs, runtimes, and verification details. Downloaded logs,
input snapshots, and the local verification report are preserved in
`oscar-production/production-20261020T160502Z/`. The source CSVs in
`data/processed/` were not changed. Scores still need to be joined to human
measurements for statistical analysis.

For Peelle joins, use the [corrected cloze table](../data/analysis/peelle.csv),
which combines 41 repeated candidate pairs using original response counts.
Collapse identical model-score rows before joining; the original score CSVs
above retain their production row counts and metadata. See the
[combination procedure and audit](../data/README.md#combining-the-peelle-repeats-for-analysis).

## DeepSeek results produced before the scoring correction

Earlier DeepSeek code scored every token of a multi-token continuation against the original sentence. The maintained scorer now appends preceding tokens and extends the attention mask. Regenerate affected multi-token DeepSeek results before using them in comparisons; the files preserved here have not been recomputed. The single-token scoring calculation is unchanged.

## Qwen results produced before the tokenization correction

Earlier Qwen preparation added a trailing space to the prompt as well as a leading space to the candidate. The corrected preparation uses one separating space and matches joint tokenization on all supplied candidate rows. Regenerate scores produced with the old Qwen preparation before comparison; historical files remain unchanged.

## Run metadata and completion

Each new score CSV has a sibling `<filename>.metadata.json` recording:

- Run ID, UTC start/end times, and status (`loading_model`, `running`, `complete`, `failed`, or `interrupted`).
- Input path/hash, cloze scale, and sentence/candidate counts.
- Requested and resolved model/tokenizer revisions, device, precision, evaluation mode, and scoring method.
- Installed package versions, Python/platform information, source-file hashes, Git commit, and whether the checkout had changes.
- Scored/skipped counts, skip reasons, and the completed output's SHA-256 hash.

No authentication tokens are included. `main` is resolved when loading the model; its resolved commit is also used for the tokenizer. If a revision cannot be resolved to a commit, the metadata reports null rather than implying it was pinned.

The runner stages scores in a temporary file and publishes them after successful inference. Invalid/nonfinite scores cause failure. Model-loading failures and handled interruptions are recorded; a hard process kill can leave a `running` record and temporary file. Treat an output as complete only when its metadata status is `complete` and its hash matches the CSV.

`--overwrite` explicitly replaces a previous run's metadata. If the replacement fails, the old score CSV is preserved, but the sidecar describes the failed new attempt and has no completed output hash. Use a new `--output` path when retaining earlier runs matters. Concurrent runs must use different output paths; do not share a path with `--overwrite`.
