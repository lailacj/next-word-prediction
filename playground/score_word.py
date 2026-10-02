"""Experiment with scoring a single continuation using the shared package."""

import argparse

from next_word_prediction import LanguageModel
from next_word_prediction.models import MODEL_IDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_IDS, default="bert")
    parser.add_argument("--sentence", default="The capital of France is")
    parser.add_argument("--word", default="Paris")
    args = parser.parse_args()

    model = LanguageModel(args.model)
    tokens = model.tokenize_word(args.word)
    score = None
    if tokens:
        score = model.predict_next_word(model.tokenize_sentence(args.sentence), tokens)
    print("Unsupported continuation" if score is None else f"Log probability: {score}")


if __name__ == "__main__":
    main()
