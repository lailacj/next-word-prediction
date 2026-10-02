# Playground

Use this directory for scratch scripts, notebooks, and small experiments. Import reusable functionality from `next_word_prediction`; promote useful code into [src/next_word_prediction/](../src/next_word_prediction/) and correctness tests into [tests/](../tests/).

After installing the package, try a single continuation from the repository root:

```sh
python playground/score_word.py --model bert --sentence "The capital of France is" --word Paris
```

This loads model weights and prints a score without writing dataset results. For full dataset runs, use the `next-word-prediction` command described in the [project README](../README.md).

Earlier experiments and standalone runners are preserved in [archive/playground/](../archive/playground/).
