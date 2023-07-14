import pandas as pd

def split_and_average(values: list) -> list:
    """

    """
    new_values = [sum(values[i:i+5]) / 5 for i in range(0, len(values), 5)]
    return new_values

if __name__ == "__main__":
    # extract the data
    data = pd.read_csv('results_f1_valid.csv')
    new_df = pd.DataFrame()

    # Iterate over column names
    for column in data:
        
        # Select column contents by column
        # name using [] operator
        values = data[column].tolist()

        if column == 'Name':
            new_values = [values[i] for i in range(0, len(values), 5)]
        else:
            new_values = [sum(values[i:i+5]) / 5 for i in range(0, len(values), 5)]
    
        new_df[column] = new_values
    
    new_df.to_csv('averaged_results_f1.csv')