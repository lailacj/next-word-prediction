#!/bin/bash
#SBATCH -J next-word-prediction-pipeline
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time 8:00:00
#SBATCH -o /users/diriho/data/diriho/next-word-prediction/batch_logs/frequency_output.%j.out
#SBATCH -e /users/diriho/data/diriho/next-word-prediction/batch_logs/frequency_err.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=don_destin_iriho@brown.edu

echo "Running the next word prediction pipeline..."

python /users/diriho/data/diriho/next-word-prediction/pipeline/run_pipeline.py