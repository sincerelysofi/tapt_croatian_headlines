import os
from transformers import AutoTokenizer
from datasets import Dataset
from collections import Counter
from headlines_analysis import get_headlines, save_distribution
from tqdm import tqdm

def tokenize_function(tokenizer,
                      data: Dataset) -> Dataset:
    """Runs the given tokenizer on the given Dataset object.

    Args:
        tokenizer (BertTokenizerFast): the tokenizer to be used
        data (Dataset): the Dataset to be processed

    Returns:
        Dataset: A dataset that has been appropriately tokenized.

    """
    return tokenizer(data['headlines'])
    

def get_dataset(path: str,
                tokenizer,
                from_preprocessed=False,
                model_name='') -> Dataset:
    """ Extracts the headlines from the given path and then runs the BERTić
    tokeniser on it to get sub-token distribution.

    Args:
        path (str): A path to the headline csv file.
        tokenizer (BertTokenizerFast): The BERT WordPiece tokenizer to use.
        from_preprocessed (:obj: `bool`, optional): Whether this dataset is
            being loaded from a preprocessed.txt file. By default, it is not.
        model_name (:obj: `str`, optional): The name of the model.

    Returns:
        None
    """
    raw_dataset = []

    # check if there is a pre-processed text file already
    if from_preprocessed:
        with open('preprocessed.txt') as f:
            raw_dataset = [h.rstrip('\n\r') for h in f.readlines()]
    else:
        raw_dataset = get_headlines(path, quiet=True)

    dataset = Dataset.from_dict({'headlines': raw_dataset})

    # tokenise the dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(tokenizer, x), batched=True,
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

    print('Done generating statistics')
    

def main() -> None:
    model_checkpoint = 'classla/bcms-bertic'
    model_name = model_checkpoint.split('/')[-1]

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    # now load the dataset
    path = 'articles_titles_all.csv'

    preprocessed_exists = os.path.exists('preprocessed.txt')

    get_dataset(path,
                tokenizer,
                from_preprocessed=preprocessed_exists,
                model_name=model_name)


if __name__ == "__main__":
    main()
