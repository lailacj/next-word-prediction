# Results

New runs default to `results/<dataset stem>/<model>.csv` relative to the working directory. Run from the repository root to save them here, or select a destination with `--output`.

[legacy/](legacy/) preserves the original `data/Model_*` directories and their CSVs unchanged. Their model-specific headers and scoring conventions may differ from future runs; directory relocation does not validate or recompute scores.

New result files are ignored by Git by default. To preserve a selected result in version control, add it explicitly with `git add -f <path>` and record how it was produced.
