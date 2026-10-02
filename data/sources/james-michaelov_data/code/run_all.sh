#!/bin/bash

for MODEL in facebook/opt-125m facebook/opt-350m facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b facebook/opt-13b facebook/xglm-564M facebook/xglm-1.7B facebook/xglm-2.9B facebook/xglm-7.5B bigscience/bloom-560m bigscience/bloom-1b1 bigscience/bloom-1b7 bigscience/bloom-3b bigscience/bloom-7b1 HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-1.7B openai-community/gpt2 openai-community/gpt2-medium openai-community/gpt2-large openai-community/gpt2-xl

do
    for CHECKPOINT in main
    do
        sbatch calculate_surprisal.sh $MODEL $CHECKPOINT
    done
done

for MODEL in EleutherAI/pythia-14m EleutherAI/pythia-31m EleutherAI/pythia-70m EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1b EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b EleutherAI/pythia-6.9b EleutherAI/pythia-12b
do
    for CHECKPOINT in step0 step1 step2 step4 step8 step16 step32 step64 step128 step256 step512 step1000 step2000 step4000 step8000 step16000 step32000 step64000 step128000 step143000
    do
        sbatch calculate_surprisal.sh $MODEL $CHECKPOINT
    done
done

for MODEL in openai-community/gpt2 openai-community/gpt2-medium openai-community/gpt2-large openai-community/gpt2-xl

do
    for CHECKPOINT in main
    do
        sbatch calculate_surprisal_truncate.sh $MODEL $CHECKPOINT
    done
done