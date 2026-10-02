#!/bin/bash


lm_eval --model hf \
    --model_args pretrained=$1,revision=$2\
    --tasks  wikitext,blimp,lambada_openai,hellaswag,piqa,winogrande\
    --device cuda:0 \
    --batch_size auto \
    --output_path eval_results