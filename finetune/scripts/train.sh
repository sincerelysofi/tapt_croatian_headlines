#!/bin/bash

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

MODELS_DIR="../../models/"

# let's assume that this models folder ONLY has models in it
if [ ! -d $MODELS_DIR ]
then
	echo Models directory not found. Exiting.
	exit 1
fi

for CHECKPOINT in ../../models/*
do
	case $CHECKPOINT in

		../../models/crosloengual-bert)
			MODEL="EMBEDDIA/crosloengual-bert"
			;;

		../../models/xlm-roberta-base)
			MODEL="xlm-roberta-base"
			;;
		
		../../models/xlm-roberta-large)
			MODEL="xlm-roberta-large"
			;;
		
		../../models/bert-base-multilingual-cased)
			MODEL="bert-base-multilingual-cased
			;;
		
		../../models/bcms-bertic)
			MODEL="classla/bcms-bertic"
			;;
	esac

	echo "Current model: ${MODEL}"
	for TRAINING_STATUS in untrained trained
	do
		for SEED in 42 43 44 45 46
		do
			if [ $TRAINING_STATUS == untrained ]
			then
				USE_CHECKPOINT=$MODEL
			else
				USE_CHECKPOINT=$CHECKPOINT
			fi
			NAME="FINAL_${MODEL}-${TRAINING_STATUS}=${SEED}"
			
			python ../evaluation/training.py \
				--lr $LR \
				--model "$MODEL" \
				--checkpoint "$USE_CHECKPOINT" \
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
	done
done
