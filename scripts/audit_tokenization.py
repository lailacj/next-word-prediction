"""Audit prepared inputs with actual tokenizers without loading model weights."""

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from next_word_prediction.datasets import load_cloze_data
from next_word_prediction.models import LanguageModel, MODEL_IDS


def audit(tokenizer, model_name, datasets):
    # Use the production preparation methods without constructing a model.
    wrapper = LanguageModel.__new__(LanguageModel)
    wrapper.name = model_name
    wrapper.tokenizer = tokenizer
    wrapper.model = SimpleNamespace(device="cpu")
    encode_word = lru_cache(maxsize=None)(wrapper.tokenize_word)
    summaries = []
    for path, records in datasets:
        summary = {
            "dataset": str(path), "candidates": 0, "single_token": 0,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "multi_token": 0, "empty": 0, "unknown": 0,
            "boundary_mismatches": None if model_name == "bert" else 0, "examples": [],
        }
        for record in records:
            context = wrapper.tokenize_sentence(record.sentence)
            if model_name != "bert":
                ids = context["input_ids"] if model_name == "deepseek" else context
                prompt_ids = ids[0].tolist()
            for candidate in record.candidates:
                summary["candidates"] += 1
                target = encode_word(candidate.word)
                if not target:
                    summary["empty"] += 1
                    continue
                summary["single_token" if len(target) == 1 else "multi_token"] += 1
                if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in target:
                    summary["unknown"] += 1
                if model_name == "bert":
                    # BERT inserts a mask and terminal period; it is not causal.
                    continue
                combined = record.sentence.strip() + " " + candidate.word.strip()
                full_ids = tokenizer.encode(combined, add_special_tokens=True)
                if prompt_ids + target != full_ids:
                    summary["boundary_mismatches"] += 1
                    if len(summary["examples"]) < 3:
                        summary["examples"].append({"sentence_id": record.sentence_id, "word": candidate.word})
        summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_IDS, default=list(MODEL_IDS))
    parser.add_argument("--datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--cloze-scale", choices=("proportion", "percent"))
    parser.add_argument("--revision", default="main", help="Tokenizer revision (use one model for a commit hash).")
    parser.add_argument("--local-files-only", action="store_true", help="Use already cached tokenizers without downloading.")
    args = parser.parse_args()

    import torch
    import transformers
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer

    datasets = [(p, load_cloze_data(p, cloze_scale=args.cloze_scale)) for p in args.datasets]
    report = {"torch": torch.__version__, "transformers": transformers.__version__, "models": {}}
    failed = False
    for name in args.models:
        model_id = MODEL_IDS[name]
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, revision=args.revision, local_files_only=args.local_files_only,
            )
        except OSError:
            report["models"][name] = {"model_id": model_id, "status": "unavailable",
                                      "message": "Tokenizer unavailable; check cache, network, and gated-model access."}
            failed = True
            continue
        cached = try_to_load_from_cache(model_id, "tokenizer_config.json", revision=args.revision)
        commit = Path(cached).parent.name if isinstance(cached, str) else None
        summaries = audit(tokenizer, name, datasets)
        mismatch = any(s["boundary_mismatches"] or s["empty"] or s["unknown"] for s in summaries)
        report["models"][name] = {
            "model_id": model_id, "requested_revision": args.revision, "resolved_revision": commit,
            "status": "mismatch" if mismatch else "ok",
            "boundary_check": "not_applicable_masked_model" if name == "bert" else "separate_vs_joint",
            "special_tokens": tokenizer.special_tokens_map, "datasets": summaries,
        }
        failed |= mismatch
    print(json.dumps(report, indent=2))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
