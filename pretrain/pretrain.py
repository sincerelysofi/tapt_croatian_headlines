import os
from sys import argv
import pickle
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForWholeWordMask,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from collections import Counter
from math import exp
from headlines_analysis import get_headlines, save_distribution
from tqdm import tqdm

def tokenize_function(tokenizer, data: Dataset, wwm=True) -> Dataset:
    """Runs the given tokenizer on the given Dataset object and also adds the
    appropriate word_ids.

    Args:
        tokenizer (BertTokenizerFast): the tokenizer to be used
        data (Dataset): the Dataset to be processed

    Returns:
        Dataset: A dataset that has been appropriately tokenized.

    """
    tok_data = tokenizer(data['headlines'])

    # add the word_ids for full word masking
    if wwm:
        tok_data['word_ids'] = [
            tok_data.word_ids(i) for i in range(len(tok_data["input_ids"]))
        ]

    return tok_data


def chunk_dataset(dataset) -> Dataset:
    """Splits the given dataset into equal-sized chunks, 512 in length.
    
    Args:
        dataset (LazyBatch): a batch of data to work with.

    Returns:
        Dataset: The dataset, which has now been split into 512-size chunks.

    """
    # combine everything into one
    combined = {k: sum(dataset[k], []) for k in dataset.keys()}

    # get the number of chunks to divide the concatenated string into
    num_chunks = len(combined['input_ids']) // 512

    # now recombine
    result = {
        k: [t[i*512:(i+1)*512] for i in range(num_chunks)]
            for k, t in combined.items()
    }

    # labels and inputs are the same, so just copy them
    result['labels'] = result['input_ids'].copy()
    
    return result


def get_dataset(path, tokenizer, from_preprocessed=False, wwm=True,
                model_name='') -> Dataset:
    """ Extracts the headlines from the given path and prepares it for training
    use.

    Args:
        path (str): A path to the headline csv file.
        tokenizer (BertTokenizerFast): The BERT WordPiece tokenizer to use.
        from_preprocessed (:obj: `bool`, optional): Whether this dataset is
            being loaded from a preprocessed.txt file. By default, it is not.
        wwm (:obj: `bool`, optional): Whether to prepare this for whole word
            masking. By default, it does, passing the argument to the
            tokenizer.
        model_name (:obj: `str`, optional): The name of the model.

    Returns:
        Dataset: The dataset, which has been processed and concatenated
            accordingly.

    """
    raw_dataset = []

    if from_preprocessed:
        with open('preprocessed.txt') as f:
            raw_dataset = [h.rstrip('\n\r') for h in f.readlines()]
    else:
        raw_dataset = get_headlines(path, quiet=True)

    dataset = Dataset.from_dict({'headlines': raw_dataset})

    # tokenise the dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(tokenizer, x, wwm), batched=True,
        remove_columns=['headlines']
    )

    # take this opportunity to show the distribution of sub-tokens
    print('Gathering statistics on subtoken distribution...')
    
    combined = []

    for piece in tqdm(tokenized_dataset['input_ids']):
        combined += piece
    
    combined_subtokens = Counter([tokenizer.decode(x) for x in tqdm(combined)])

    # save the distribution for later analysis
    save_distribution(combined_subtokens,
                      f'subtoken_count_{model_name + "_"}' + path.replace('csv', 'tsv'),
                      'subtoken\tcount')

    # now split the dataset into chunks
    dataset = tokenized_dataset.map(chunk_dataset, batched=True)
    
    print('Dataset successfully concatenated.')

    return dataset


def get_perplexity(trainer) -> float:
    """Performs evaluation, reports and then returns perplexity.

    Args:
        trainer (Trainer): A trainer object to run evaluation on.

    Returns:
        float: Perplexity (loss) to be reported.

    """
    eval_results = trainer.evaluate()

    print(f">>> Perplexity: {exp(eval_results['eval_loss']):.2f}")
    
    return exp(eval_results['eval_loss'])


def split_dataset(dataset: Dataset, divisor=100) -> tuple[Dataset, Dataset]:
    """Splits a dataset into train and evaluation sets.
    
    Returns two datasets, one for training, one for evaluation. By default, the
    data set is split into 99 training / 1 eval, but can be set to any other
    amount by adjusting the divisor parameter.

    Args:
        dataset (Dataset): A dataset
        divisor (:obj: `int`, optional): The divisor for the split.

    Returns:
        Dataset: The training dataset
        Dataset: The evaluation dataset
    """
    # shuffle the dataset, as the datasets tend to be sorted..
    dataset = dataset.shuffle(seed=1)
    size = len(dataset)

    # by default, a 99/1 split; set divisor to change
    print(f'Splitting dataset of size {size}...')

    train_dataset = dataset.select(range(size // divisor, size))
    eval_dataset = dataset.select(range(size // divisor))

    print('Training size:', len(train_dataset))
    print('Evaluation size:', len(eval_dataset))

    return train_dataset, eval_dataset


def main() -> None:
    # load the model first
    args = argv[1:]
    wwm = True

    if len(args) == 2 and args[0] == '-model':
        model_checkpoint = args[1]
        collator = DataCollatorForLanguageModeling
        wwm = False
    else:
        print('Either no model was specified or invalid syntax was used.')
        print('Using default CroSloEngual-BERT...')
        model_checkpoint = 'EMBEDDIA/crosloengual-bert'
        collator = DataCollatorForWholeWordMask

    model_name = model_checkpoint.split('/')[-1]

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # initialise the collator
    data_collator = collator(tokenizer)

    # now load the dataset
    path = 'articles_titles_all.csv'

    # check for an existing pickle
    model_pickle = f'headlines_{model_name}.pkl'

    if os.path.exists(model_pickle):
        # load the pickle
        print(f'>>> model_pickle found! loading...')
        with open(model_pickle, 'rb') as f:
            dataset = pickle.load(f)
    else:
        # check if there's a preprocessed dataset already and save some time...
        preprocessed_exists = os.path.exists('preprocessed.txt')

        if preprocessed_exists:
            print('>>> preprocessed.txt found! generating dataset with this file...')
        else:
            # generate a fresh dataset
            print('>>> Generating dataset...')
        
        dataset = get_dataset(path,
                              tokenizer,
                              from_preprocessed=preprocessed_exists,
                              wwm=wwm,
                              model_name=model_name)

        print(f'>>> Done! Now saving to {model_pickle}...')

        with open(model_pickle, 'wb') as f:
            pickle.dump(dataset, f)

    # split the dataset
    train_dataset, eval_dataset = split_dataset(dataset)

    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    # initiate the arguments
    args = TrainingArguments(
        output_dir="output" + model_name, # save on lovorka
        num_train_epochs=3,
        save_total_limit=3,
        remove_unused_columns=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        evaluation_strategy='steps'
    )

    # prepare the trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    # get the initial perplexity
    print('getting before-training perplexity:')
    pre_per = get_perplexity(trainer)

    # perform training!
    trainer.train()

    # compare the perplexity afterwards to ensure that training was successful.
    print('getting post-training perplexity:')
    post_per = get_perplexity(trainer)

    # if there's a decreate in perplexity, then we can say it was successful!
    if post_per < pre_per:
        print(f'>>> Perplexity decrease of {pre_per-post_per}.')
        print('>>> Training successful. Saving model.')

        # save the model and report success
        trainer.save_model(model_name)

        print('All operations performed successfully!')

    else:
        # we should never see this, but perhaps something very wrong happened
        print('>>> Oops, perplexity increased... Something may have gone wrong.')


if __name__ == "__main__":
    main()
