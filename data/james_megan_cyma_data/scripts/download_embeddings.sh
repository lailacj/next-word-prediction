#!/bin/bash

wget "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip" -O ../data/crawl-300d-2M.vec.zip --show-progress
unzip ../data/crawl-300d-2M.vec.zip -d ../data
rm ../data/crawl-300d-2M.vec.zip

wget "https://nlp.stanford.edu/data/glove.840B.300d.zip" -O ../data/glove.840B.300d.zip --show-progress
unzip ../data/glove.840B.300d.zip -d ../data
rm ../data/glove.840B.300d.zip