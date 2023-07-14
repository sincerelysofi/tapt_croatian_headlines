#!/bin/bash

# This file is for quick training of **one** model with the given seeds.

# these are constant so we are separating them
LR=0.00001
DO_LOWER_CASE=False
DATASET_TRAIN_PATH="../data/processed/stone_gold_label_train.csv"
DATASET_VALID_PATH="../data/processed/stone_gold_label_valid.csv"
DATASET_TEST_PATH="../data/processed/stone_gold_label_test.csv"
EPOCHS=10
BATCH_SIZE=16
DEVICE="cuda:0"
EVAL_MODE="macro"
SCHEDULER_WARMUP_STEPS=0

# chage this to whatever base you are using
MODEL="bert-base-multilingual-cased"

# this should be the path to a trained model.
CHECKPOINT="../../models/${MODEL}"

# otherwise, uncomment to use an un-pretrained model!
#CHECKPOINT="$MODEL"

TRAINING_STATUS=trained

echo "Current model: ${MODEL}"

for SEED in 42 43 44 45 46
do
	NAME="${MODEL}-${TRAINING_STATUS}=${SEED}"
	
	python ../baselines/training.py \
		--lr $LR \
		--model "$MODEL" \
		--checkpoint "$CHECKPOINT" \
		--do_lower_case $DO_LOWER_CASE \
		--dataset_train_path "$DATASET_TRAIN_PATH" \
		--dataset_valid_path "$DATASET_VALID_PATH" \
		--dataset_test_path "$DATASET_TEST_PATH" \
		--epochs $EPOCHS \
		--batch_size $BATCH_SIZE \
		--name "$NAME" \
		--seed $SEED \
		--device "$DEVICE" \
		--eval_mode "$EVAL_MODE" \
		--scheduler_warmup_steps $SCHEDULER_WARMUP_STEPS
done
