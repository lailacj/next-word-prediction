# Results

New runs default to `results/<dataset stem>/<model>.csv` relative to the working directory. Run from the repository root to save them here, or select a destination with `--output`.

[legacy/](legacy/) preserves the original `data/Model_*` directories and their CSVs unchanged. Their model-specific headers and scoring conventions may differ from future runs; directory relocation does not validate or recompute scores.

New result files are ignored by Git by default. To preserve a selected result in version control, add it explicitly with `git add -f <path>` and record how it was produced.

## DeepSeek results produced before the scoring correction

Earlier DeepSeek code scored every token of a multi-token continuation against the original sentence. The maintained scorer now appends preceding tokens and extends the attention mask. Regenerate affected multi-token DeepSeek results before using them in comparisons; the files preserved here have not been recomputed. The single-token scoring calculation is unchanged.
