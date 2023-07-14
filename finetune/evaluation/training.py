# author: ddukic

import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import wandb
from constants import *
from dataset import SentimentDatasetVanilla
from torch.utils import data
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from util import compute_metrics_multitask, logging_multitask

from models import (
    TLBertSentimentSequenceClassification,
    XLMSentimentSequenceClassification,
    BerticSentimentSequenceClassification,
)

# Ensure reproducibility on CUDA
# set a debug environment variable CUBLAS_WORKSPACE_CONFIG to ":16:8" (may limit overall performance)
# or ":4096:8" (will increase library footprint in GPU memory by approximately 24MiB)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


def parse_boolean(value):
    return True if value.lower() in ["true", "yes", "y", "1", "t"] else False


parser = argparse.ArgumentParser(
    description="Runs training for sentiment classification tasks"
)
parser.add_argument("--lr", type=float, help="Learning rate for optimizer")
parser.add_argument(
    "--model",
    type=str,
    help="Hugging face model identifier (for now 'bert-base-cased' or 'roberta-base')",
)
parser.add_argument(
    '--checkpoint',
    type=str,
    help="Path to a folder containing the trained version of the model"
)
parser.add_argument(
    "--do_lower_case",
    type=parse_boolean,
    help="Transformer tokenizer option for lowercasing",
)
parser.add_argument(
    "--dataset_train_path",
    type=str,
    help="The train dataset path",
)
parser.add_argument(
    "--dataset_valid_path",
    type=str,
    help="The valid dataset path",
)
parser.add_argument(
    "--dataset_test_path",
    type=str,
    help="The test dataset path",
)
parser.add_argument("--epochs", type=int, help="Number of epochs")
parser.add_argument("--batch_size", type=int, help="Size of a batch")
parser.add_argument(
    "--name",
    type=str,
    help="Name of experiment on W&B",
)
parser.add_argument(
    "--seed",
    type=int,
    help="Seed to ensure reproducibility by using the same seed everywhere",
)
parser.add_argument("--device", type=str, help="GPU device to run training on")
parser.add_argument(
    "--eval_mode", type=str, help="Evaluation mode: 'micro' or 'macro'"
),
parser.add_argument(
    "--scheduler_warmup_steps",
    type=int,
    help="Number of warmup steps for scheduler (0 or more)",
)
args = parser.parse_args()

# gotta count add_argument calls
all_passed = sum([v is not None for k, v in vars(args).items()]) == len(vars(args))

print("All arguments passed?", all_passed)

if not all_passed:
    exit(1)

wandb.init(
    project="retriever_sentiment",
    entity="sofialee",
    name=args.name,
    config=args,
)

config = wandb.config

wandb.define_metric("epoch")
wandb.define_metric("train/epoch*", step_metric="epoch")
wandb.define_metric("valid/epoch*", step_metric="epoch")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(
    model,
    loader_train,
    loader_valid,
    optimizer,
    criterion,
    scheduler,
    epochs,
    eval_mode,
):

    best_valid_score = 0.0
    best_model = copy.deepcopy(model.state_dict())

    for i in tqdm(range(epochs)):
        train_loss = calculate_epoch_loss(
            model=model,
            loader=loader_train,
            optimizer=optimizer,
            criterion=criterion,
            mode="train",
        )

        scheduler.step()

        valid_loss = calculate_epoch_loss(
            model=model,
            loader=loader_valid,
            optimizer=optimizer,
            criterion=criterion,
            mode="valid",
        )

        train_all_metrics_sentiment = logging_multitask(
            model=model,
            loader=loader_train,
            loss=train_loss,
            epoch=i,
            mode="train",
            eval_mode=eval_mode,
        )

        valid_all_metrics_sentiment = logging_multitask(
            model=model,
            loader=loader_valid,
            loss=valid_loss,
            epoch=i,
            mode="valid",
            eval_mode=eval_mode,
        )

        # x axis goes from 0 to 9 because of wandb
        train_metrics = {
            "train/epoch_loss_avg": train_loss,
            "train/epoch_f1_score_sentiment": train_all_metrics_sentiment["f1"],
            "epoch": i,
        }

        valid_metrics = {
            "valid/epoch_loss_sentiment": valid_loss,
            "valid/epoch_f1_score_sentiment": valid_all_metrics_sentiment["f1"],
            "epoch": i,
        }

        wandb.log({**train_metrics, **valid_metrics})

        valid_f1 = valid_all_metrics_sentiment["f1"]

        if valid_f1 > best_valid_score:
            best_model = copy.deepcopy(model.state_dict())
            best_valid_score = valid_f1

    return best_model, best_valid_score


def collect_losses(model, criterion, batch, task="sentiment"):
    """

    """
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        ner_ids=batch["ner_ids"],
        task=task,
    )

    labels_key = "sentiment_labels" 

    # model.num_labels is equal to out.logits.shape[-1]
    # the same number of labels applies for both tasks
    tag_loss = criterion(
        out.logits.view(-1, model.num_labels), batch[labels_key].view(-1)
    )

    return tag_loss


def calculate_epoch_loss(model,
                         loader,
                         optimizer,
                         criterion,
                         mode="train") -> float:
    """

    """
    loss = 0.0

    if mode == "train":
        model.train()

        loss_sentiment_total = 0.0

        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model.zero_grad()
            optimizer.zero_grad()

            # gather losses
            sentiment_loss = collect_losses(model, criterion, batch, "sentiment")
            sentiment_loss.backward()
            loss_sentiment_total += sentiment_loss.item() * batch["input_ids"].shape[0]
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        loss = loss_sentiment_total / len(loader.dataset)

    else:
        model.eval()

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}

                sentiment_loss = collect_losses(model, criterion, batch, "sentiment")

                loss += sentiment_loss.item() * batch["input_ids"].shape[0]

        loss = loss / len(loader.dataset)

    return loss


def train_evaluate(
    model,
    loader_train,
    loader_valid,
    loader_test,
    optimizer,
    criterion,
    scheduler,
    epochs,
    eval_mode,
):
    best_model, best_valid_score = train(
        model=model,
        loader_train=loader_train,
        loader_valid=loader_valid,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=epochs,
        eval_mode=eval_mode,
    )

    model.load_state_dict(best_model)

    all_metrics_sentiment = compute_metrics_multitask(
        model=model,
        loader=loader_test,
        task="sentiment",
        eval_mode=eval_mode
    )
    
    wandb.summary["valid_f1_sentiment"] = best_valid_score
    wandb.summary["test_all_metrics_sentiment"] = all_metrics_sentiment

    print("test_all_metrics_sentiment", all_metrics_sentiment)


if __name__ == "__main__":
    set_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        #config.model if 'bertic' not in config.model else config.checkpoint,
        config.model,
        do_lower_case=config.do_lower_case,
    )

    senti2id, id2senti = {}, {}

    for sentiment in SENTIMENT_CLASSES:
        id2senti[len(id2senti)] = sentiment
        senti2id[sentiment] = len(senti2id)

    # load the corresponding classifier
    if 'xlm' in config.model:
        model = XLMSentimentSequenceClassification
    elif 'bertic' in config.model:
        model = BerticSentimentSequenceClassification
    else:
        model = TLBertSentimentSequenceClassification
    
    bert_model = model.from_pretrained(
        config.checkpoint,
        num_labels=len(senti2id),
        id2labels={"sentiment": id2senti},
    )

    # pad to max sequence in batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    dataset_train, dataset_valid, dataset_test = (
        SentimentDatasetVanilla(
            fpath=fpath,
            tokenizer=tokenizer,
            senti2id=senti2id,
        )
        for fpath in [config.dataset_train_path,
                      config.dataset_valid_path,
                      config.dataset_test_path]
    )

    loader_train, loader_valid, loader_test = (
        data.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            collate_fn=data_collator,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(config.seed),
        )
        for dataset in [dataset_train, dataset_valid, dataset_test]
    )

    device = config.device if torch.cuda.is_available() else "cpu"

    print(device)

    # prepare model for training
    bert_model = bert_model.to(device)

    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=config.lr)

    # notice reduction argument
    criterion = nn.CrossEntropyLoss(reduction="mean")

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.scheduler_warmup_steps,
        num_training_steps=config.batch_size * config.epochs,
    )

    train_evaluate(
        model=bert_model,
        loader_train=loader_train,
        loader_valid=loader_valid,
        loader_test=loader_test,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=config.epochs,
        eval_mode=config.eval_mode,
    )

    wandb.finish()
