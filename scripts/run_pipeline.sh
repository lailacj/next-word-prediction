#!/bin/bash
#SBATCH -J next-word-prediction-pipeline
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time 20:00:00
#SBATCH -o next-word-prediction.%j.out
#SBATCH -e next-word-prediction.%j.err

set -euo pipefail

python -m next_word_prediction "$@"
