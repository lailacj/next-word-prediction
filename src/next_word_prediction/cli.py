"""Score a cloze CSV with a model selected from the command line."""

import argparse
from pathlib import Path

from .models import MODEL_IDS
from .pipeline import run_pipeline

MODEL_NAMES = tuple(MODEL_IDS)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to a prepared cloze CSV.")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument(
        "--output", type=Path,
        help="Output CSV (default: <working directory>/results/<dataset stem>/<model>.csv).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output CSV.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output or Path.cwd() / "results" / args.dataset.stem / f"{args.model}.csv"
    try:
        scored, skipped = run_pipeline(
            args.dataset, args.model, output, overwrite=args.overwrite,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Wrote {scored} scores to {output}; skipped {skipped} candidates.")


if __name__ == "__main__":
    main()
