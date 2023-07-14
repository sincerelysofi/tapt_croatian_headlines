import pandas as pd

def convert_data(x: str) -> None:
    """Loads the specified csv file (portion of the SToNe data set) and then
    converts everything to the processed format with just sentiment as needed
    for final use.

    Args:
        x (str): The 

    Returns:
        None
    """
    df = pd.read_csv('raw/' + x)

    df_filtered = df[
        [
            "document_id",
            "text",
            "ner",
            "ner_type",
            "ner_begin",
            "ner_end",
            "aggregated_sentiment"
        ]
    ]

    df_filtered = df_filtered.rename(
        columns = {
            "aggregated_sentiment": "sentiment"
            }
    )

    df_filtered.to_csv('processed/' + x)

if __name__ == "__main__":
    for x in ["train", "valid", "test"]:
        convert_data(f'stone_gold_label_{x}.csv')
