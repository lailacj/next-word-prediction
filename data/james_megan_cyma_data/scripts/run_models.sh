#!/bin/bash

python get_gpt3_surprisal.py -i ../data/full_stims.stims -o ../data -m davinci -k YOUR_API_KEY_HERE

python get_cosine_distances.py -i ../data/full_stims.stims -o ../data -m ../data/crawl-300d-2M.vec -cs

python get_cosine_distances.py -i ../data/full_stims.stims -o ../data -m ../data/glove.840B.300d.txt -cs

python get_word_freqs.py

python get_cosine_distances.py -i ../data/bc_critical_words.txt -o ../data -m ../data/crawl-300d-2M.vec

python get_cosine_distances.py -i ../data/bc_critical_words.txt -o ../data -m ../data/glove.840B.300d.txt
