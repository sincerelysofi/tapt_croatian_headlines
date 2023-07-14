from collections import Counter
from pandas import read_csv

import seaborn as sns
import matplotlib.pyplot as plt
import reldi_tokeniser

from fuzzywuzzy import fuzz
from tqdm import tqdm


def save_distribution(data: Counter, path: str, header: str) -> None:
    """Save a counter object at a given path with the given headers. Also sorts
    the values before saving.

    Args:
        data (Counter): A Counter object to save.
        path (str): A path to save to.
        header (str): The header to put in the beginning of the file.
    
    Returns:
        None
        
    """
    sorted_data = data.most_common()

    with open(path, 'w') as f:
        f.write(f'{header}\n')

        for x in tqdm(sorted_data):
            f.write(f'{x[0]}\t{x[1]}\n')


def get_info(headlines: list) -> None:
    """Performs basic tokenisation and then report statistics about the
    headlines dataset. This includes character count, token count, and
    distribution of token count.

    Args:
        headlines (list): A list of head lines
    
    Returns:
        None

    """

    # some quick data
    char_counts = [len(h) for h in headlines]

    tokenized_headlines = [reldi_tokeniser.run(h, 'hr', bert=True) for h in tqdm(headlines)]
    
    token_counts = [len(h.split()) for h in tokenized_headlines]

    min_char, max_char = min(char_counts), max(char_counts)
    min_tok, max_tok = min(token_counts), max(token_counts)

    total = sum(token_counts)
    avg = total / len(token_counts)

    print(f'shortest headline: {min_char} chars, longest: {max_char} chars')
    print(f'shortest headline: {min_tok} tokens, longest: {max_tok} tokens')

    print(f'In total, there are {total} tokens with an average of {avg} per headline.')
    
    print('>>> Now generating a chart of the distribution of word counts...')
    distr = dict(Counter(token_counts))

    data = {
        'Number of headlines': [distr[k] for k in distr],
        'Number of tokens': [k for k in distr]
    }

    # generate and show a chart of the distribution
    sns.scatterplot(data=data,
                    x='Number of tokens',
                    y='Number of headlines').set(title='Distribution of count')
    
    # logarithmic scale
    plt.yscale('log')
    plt.savefig('./headline_token_distribution.png')

    print('>>> Chart generated and saved.')
    print('>>> Now performing an analysis of token frequency...')

    token_use = Counter()

    for h in tqdm(tokenized_headlines):
        tokens = h.split()
        for t in tokens:
            token_use[t] += 1

    save_distribution(token_use, 'word_count.tsv', header='word\tcount')
    print('>>> Analysis done. Results saved to word_count.tsv...')


def partial_match(headline: str, sorted_list_to_filter: list) -> int:
    """Given a headline and a sorted list, check if the given headline is in the
    dataset. It does this by looking for items of similar length (between 80%
    and 120%) and then performs a fuzzy match of 90% or greater. If it reaches
    a point where items are now 120%
    
    If there is a match, it returns the index of the item in the list.
    Otherwise, it returns 0.

    Args:
        headline (str): A headline to search for.
        sorted_list_to_filter (list): A sorted list to filter through. This
            list is sorted by length.

    Returns:
        int: The index number of the item if found in the list, otherwise -1.

    """
    length = len(headline)

    for i, h in enumerate(sorted_list_to_filter):
        # don't perform a match if it's much shorter than the target headline
        if length*.8 > len(h):
            continue
        # also end the search if the length is dramatically longer
        elif length*1.2 < len(h):
            return -1
        
        # perform a partial match
        if fuzz.ratio(headline, h) > 90:
            return i

    # if we somehow come to the end without any results, report no matches
    return -1


def destone(headlines: list) -> list:
    """Removes headlines from the SToNe test and validation set. The purpose of
    this function is to ensure that duplicates from SToNe are removed for the
    eventual training phase and later evaluation. This function also reports
    the number of removed headlines and offers to save the list of any headline
    that did not get removed.
    
    Args:
        headlines (list): All the headlines to be removed.

    Returns:
        list: The headlines with all the SToNe test and validation entries
            removed.

    """
    # get a count from before the removal
    pre_stone_removal = len(headlines)

    # get the stone dataset 
    stone_test = read_csv('stone_test.csv')
    stone_valid = read_csv('stone_valid.csv')

    # concatenate it
    stone = stone_valid['text'].tolist() + stone_test['text'].tolist()

    print('checking for headlines to remove from the SToNe database...')

    ### determine which headlines need to be removed
    to_destone = [h for h in tqdm(stone) if h in headlines]

    # make a list of everything in SToNe that wasn't a perfect match
    stone = [h for h in stone if h not in to_destone]
    stone.sort(key=len)

    ### filter out perfect matches
    print('filtering out headlines now...')
    headlines = [h for h in tqdm(headlines) if h not in to_destone]

    # we are sorting by length because our 
    headlines.sort(key=len)

    print('now checking for fuzzy matches...')
    new_headlines = []

    ### now filter out partial matches 
    for h in tqdm(headlines):
        result = partial_match(h, stone)

        # if a result is found, remove it from the stone list and do not add
        if result != -1:
            removed = stone.pop(result)

            print(f'Fuzzy match found:\n{removed}\n{h}')
            print('Remaining count of SToNe headlines:', len(stone))

        else:
            # no match? then add it to the new list
            new_headlines.append(h)

    headlines = new_headlines

    destoned = pre_stone_removal - len(headlines)
    print(f'{destoned} headlines from SToNe dataset removed')

    # let's verify whether any headlines were not removed
    leftover = len(stone)

    if leftover:
        print(f'{leftover} headline(s) were not removed...')
        print('Listing all unremoved headlines...')

        # list all the headlines
        for i, h in enumerate(stone):
            print(f'{i+1}. {h}')
        
        # now prompt the user whether they want to save the headlines/continue
        match input('Continue? (Y/n/[s]ave)'):
            case 'n':
                print('Exiting...')
                exit()
            case 's':
                dump_list('stone_unremoved.txt', stone)
                
                print('Unused headlines written to stone_unremoved.txt')
                print('Continuing...')
            case _:
                print('Continuing...')

    return headlines


def possible_fuzzy(text: str, index: int, dataset: list) -> bool:
    """Given a sorted dataset, text and index of that text in the dataset, the
    text is compared with the next item in the sorted dataset. If the text is
    either a sub-text of the subsequent or matches at least 90% of the text,
    return True (a fuzzy match was found). Otherwise, return False.

    Args:
        text (str):
        index (int):
        dataset (list):
    
    Returns:
        bool: Whether a fuzzy match was found.

    """
    # check that we're not at the end of the list
    if index == len(dataset)-1:
        return False
    
    # grab the next item for comparison
    next_item = dataset[index+1]

    # if current item is within the next item or there's a 90% fuzzy match,
    # report the match
    if text in next_item or fuzz.ratio(text, next_item) > 90:
        print(f'Adjacent fuzzy match found:\n{text}\n{next_item}')
        return True

    else:
        return False


def defuzz(headlines: list) -> list:
    """Recursively remove fuzzy matches. Essentially, it uses a sorted list
    and scans through the list, comparing each item with the subsequent item.
    If the current item is NOT a substring of the subsequent item NOR a fuzzy
    match, then the item is added to the new headlines dataset. This is
    preformed recursively until no matches are found.

    After that, the number of passes performed is reported and the deduplicated
    headlines dataset is returned.

    Args:
        headlines (list): An alphabetically sorted list of headlines.

    Returns:
        list: The deduplicated headlines

    """
    matches_found = True
    passes = 0

    # perform this until there are no longer fuzzy matches
    while matches_found:
        new_headlines = []
        matches = 0

        # scan through the entire list
        for i, h in enumerate(headlines):
            if not possible_fuzzy(h, i, headlines):
                # add it if it's not a match
                new_headlines.append(h)
            else:
                # don't add it if it is a match; report the match
                matches += 1
        
        headlines = new_headlines
        passes += 1

        # check if there are any matches left
        matches_found = (matches > 0)
    
    print(f'{passes} round(s) of fuzzy matching performed.')

    return headlines


def dump_list(path, a_list) -> None:
    """Dumps a list into a text file.

    Args:
        path (str): The path to save the file at.
        a_list (list): The list to dump.
    
    Returns:
        None

    """
    with open(path, 'w') as f:
        for i in a_list:
            f.write(i + '\n')


def preprocess(headlines, remove_stone=True) -> list:
    """Performs four stages of preprocessing. The first stage removes all
    exact duplicates. The second stage prunes one-word headlines for whole word
    masking. The third stage recursively removes adjacent fuzzy matches. Then,
    the fourth stage, if requested, will remove entries that are used in the
    SToNe evaluation and testing dataset.

    The preprocessed headlines list is then returned.

    Args:
        headlines (list): The dataset on which preprocessing will be performed.
        remove_stone (:obj: `bool`, optional): Whether to remove the SToNe
            dataset lines. Will be performed by default.
    
    Returns:
        list: The headlines, now deduplicated and sorted.

    """

    ### first stage: exact matches
    prededupe = len(headlines)

    headlines = list(set(headlines))
    dedupe = len(headlines)

    print(f'>>> {prededupe-dedupe} headlines removed from first round of deduplication')

    ### second stage, prune one-word headlines
    headlines = [h for h in headlines if len(h.split()) > 1]
    print(f'>>> {dedupe-len(headlines)} one-word headlines pruned')

    ### third stage, sort the headlines and perform fuzzy removal
    print('Now sorting headlines for fuzzy algorithm...')
    headlines.sort()
    
    print('Performing fuzzy removal...')
    prefuzz = len(headlines)
    headlines = defuzz(headlines)
    print(f'>>> {prefuzz-len(headlines)} headlines removed during second round of deduplication)')

    ### fourth stage, remove lines from SToNe dataset
    if remove_stone:
        headlines = destone(headlines)
    
    print(f'>>> Final dataset contains {len(headlines)} headlines')
    
    return headlines


def get_headlines(text: str, quiet=False, remove_stone=True) -> list:
    """This extracts all the headlines from a given csv file and returns it as
    a deduplicated list.
    
    Args:
        text (str): The path to the headlines dataset.
        quiet (:obj: `boolean'): Whether to display statistical information
            about the dataset.
        remove_stone (:obj: `boolean'): Whether to remove entries from the
            SToNe dataset.

    Returns:
        list: All the headlines, deduplicated.

    """
    # load the headlines file
    alldata = read_csv(text)
    headlines = alldata['title'].tolist()

    print(f'>>> loaded {len(headlines)} headlines')

    # perform deduplication and remove the stone dataset if applicable
    headlines = preprocess(headlines, remove_stone)

    # show some information about the dataset if wanted 
    if not quiet:
        print('>>> Saving portal distribution...')
        portal_distr = Counter(alldata['portal'].tolist())
        save_distribution(portal_distr, 'portal_distr.tsv', header='name\tcount')
        print('>>> Done!')

        print('>>> Now displaying information about headlines...')
        get_info(headlines)

    return headlines


def main():
    headlines = get_headlines('articles_titles_all.csv')

    # if this is run directly, then give the option to save it
    print('Analysis complete. Save as pre-processed headline set? (y/N)')

    if input() == 'y':
        dump_list('preprocessed.txt', headlines)
        print('Contents dumped to preprocessed.txt.')


if __name__ == "__main__":
    main()