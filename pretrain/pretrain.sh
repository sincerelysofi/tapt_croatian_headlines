#!/bin/bash

models="bert-base-multilingual-cased xlm-roberta-base xlm-roberta-large"

echo Pre-training cseBERT...
torchrun pretrain.py
echo Done!

for m in $models
do
	echo Pre-training $m ...
	torchrun pretrain.py -model $m
	echo Done!
done

echo Now moving onto BERTić...
python get_token_distribution_bertic.py
python bertic_pretrain.py
echo Done!