"""Score a cloze CSV with a model selected from the command line."""

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("qwen", "bert", "deepseek", "llama")


def create_model(name):
    # Import only when running inference so --help needs no ML dependencies.
    if __package__:
        from .language_models import BertModel, DeepSeekModel, LlamaModel, QwenModel
    else:
        from language_models import BertModel, DeepSeekModel, LlamaModel, QwenModel

    model_classes = {
        "qwen": QwenModel,
        "bert": BertModel,
        "deepseek": DeepSeekModel,
        "llama": LlamaModel,
    }
    return model_classes[name]()


def load_dataset(path):
    if __package__:
        from .data_organization import load_cloze_data
    else:
        from data_organization import load_cloze_data

    return load_cloze_data(path)


def run_pipeline(dataset, model_name, output, *, overwrite=False):
    """Write scores and return the number of scored and skipped candidates."""
    dataset = Path(dataset)
    output = Path(output)
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset}")
    if dataset.resolve() == output.resolve():
        raise ValueError("The output path must differ from the dataset path.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")

    masked, sentences = load_dataset(dataset)
    model = create_model(model_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored = skipped = 0

    # Exclusive creation also prevents overwriting a file created during model loading.
    with output.open("w" if overwrite else "x", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sentence_num", "sentence", "word", f"{model_name}_prob"])
        for (sentence_id, candidates), sentence in zip(masked.items(), sentences):
            sentence_token_ids = model.tokenize_sentense(sentence)
            for word, _cloze_prob in candidates:
                word_token_ids = model.tokenize_word(word)
                if not word_token_ids:
                    skipped += 1
                    continue
                score = model.predict_next_word(sentence_token_ids, word_token_ids)
                if score is None:
                    skipped += 1
                    continue
                writer.writerow([sentence_id, sentence, word, score])
                scored += 1

    return scored, skipped


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to a prepared cloze CSV.")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument(
        "--output", type=Path,
        help="Output CSV (default: <project>/results/<dataset stem>/<model>.csv).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output CSV.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output or PROJECT_ROOT / "results" / args.dataset.stem / f"{args.model}.csv"
    try:
        scored, skipped = run_pipeline(
            args.dataset, args.model, output, overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Wrote {scored} scores to {output}; skipped {skipped} candidates.")


if __name__ == "__main__":
    main()
