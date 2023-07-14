# author: ddukic
# edited by sofialee for Retriever Sentimetn analysis 

import pandas as pd
from torch.utils import data


class SentimentDatasetVanilla(data.Dataset):
    def __init__(self, fpath, tokenizer, senti2id):
        data = pd.read_csv(fpath)

        self.tokenized_inputs = self.tokenize_data(data, tokenizer, senti2id)

    def __len__(self):
        return len(self.tokenized_inputs["input_ids"])

    def __getitem__(self, idx):
        return {
            k: self.tokenized_inputs[k][idx]
            for k in [
                "input_ids",
                "attention_mask",
                "ner_ids",
                "sentiment_labels",
                "document_ids",
            ]
        }

    def tokenize_data(self, data, tokenizer, senti2id):
        # extract the text and spans
        texts, ner_starts, ner_ends = (
            data["text"].tolist(),
            data["ner_begin"].tolist(),
            data["ner_end"].tolist(),
        )

        # get the labels
        sentiment_labels = data["sentiment"].tolist()

        self.texts = texts
        self.ner_starts = ner_starts
        self.ner_ends = ner_ends
        self.sentiment_labels = sentiment_labels

        tokens = tokenizer(texts, truncation=True)

        start_positions = []
        end_positions = []

        for i in range(len(texts)):
            start_positions.append(tokens.char_to_token(i, ner_starts[i]))
            end_positions.append(tokens.char_to_token(i, ner_ends[i] - 1))

            # if start position is None, the text has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length

            # if end position is None, the 'char_to_token' function points to the space before the correct token - > add + 1
            if end_positions[-1] is None:
                end_positions[-1] = tokens.char_to_token(i, ner_ends[i])

        tokens["ner_ids"] = list(zip(start_positions, end_positions))
        tokens["sentiment_labels"] = [senti2id[x] for x in sentiment_labels]
        tokens["document_ids"] = data["document_id"].tolist()

        return tokens
