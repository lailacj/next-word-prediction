#!/bin/bash


python calculate_surprisal.py -o ../results -m $1 -c $2 -ii stimuli.txt  --dtype float32 --eos_as_bos
