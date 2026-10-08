#!/bin/bash
# Run all prepared datasets with each supported model, one process at a time.
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
    echo 'Usage: bash scripts/run_all.sh [--device cpu|cuda|cuda:N|mps] [--dtype float32|float16|bfloat16] [--overwrite]'
    echo 'Run from the repository root with the package environment activated.'
    exit 0
fi

# These options are forwarded identically to every run; dataset/model/revision
# selection belongs to the individual-run CLI instead.
options=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device|--dtype)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            options+=("$1" "$2")
            shift 2
            ;;
        --overwrite)
            options+=("$1")
            shift
            ;;
        *) echo "Unsupported batch option: $1" >&2; exit 2 ;;
    esac
done

for dataset in michaelov_2024 nieuwland_2018 szewczyk_2022 peelle; do
    [[ -f "data/processed/$dataset.csv" ]] || { echo "Missing data/processed/$dataset.csv" >&2; exit 1; }
done

for model in bert deepseek qwen llama; do
    for dataset in michaelov_2024 nieuwland_2018 szewczyk_2022 peelle; do
        python -m next_word_prediction --dataset "data/processed/$dataset.csv" --model "$model" ${options[@]+"${options[@]}"}
    done
done
