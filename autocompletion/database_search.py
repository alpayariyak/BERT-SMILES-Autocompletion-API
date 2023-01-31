
def get_database_autocompletions(input_smiles, database_df, n=5):
    # Filter the dataframe to only include molecules that contain the input molecule's SMILES string as a substring
    extended_df = database_df[database_df['SMILES'].str.startswith(input_smiles)]
    # Sort by length
    sorted_df = extended_df.sort_values(by='SMILES', key=lambda x: x.str.len())

    sorted_list = sorted_df['SMILES'].head(n).tolist()
    return sorted_list


