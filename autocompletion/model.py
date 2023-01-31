from transformers import AutoModelForSequenceClassification
import pandas as pd
from rdkit import Chem
import torch
from rdkit import RDLogger
import math

RDLogger.DisableLog('rdApp.*')


def is_valid(smiles):
    return Chem.MolFromSmiles(smiles) is not None


def to_canonical_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def load_data(path):
    def clean_df(df):
        df = df.dropna()
        df = df[df['SMILES'].apply(is_valid)]
        df = df.dropna()
        df = df.drop_duplicates(subset=['SMILES'])
        df = df.reset_index(drop=True)
        return df

    # Read in the data
    database_df = pd.read_csv(path)
    # Limit SMILES to 50 characters
    database_df = database_df.loc[database_df['SMILES'].str.len() <= 50][['SMILES', 'NAME']]

    database_df = clean_df(database_df)
    database_df['SMILES_tokenized'] = database_df['SMILES'].apply(smi_tokenizer)

    return database_df


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return tokens


def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        'alpayariyak/SMILES_Next_Element_Classifier', num_labels=56)
    model.eval()
    return model


def load_tokenizer(data_path):
    class ElementTokenizer:
        def __init__(self, mapping, vocab_size=56, sequence_length=8, pad_token=0):
            self.mapping = mapping
            self.inverse = {v: k for k, v in mapping.items()}
            self.vocab_size = vocab_size
            self.sequence_length = sequence_length
            self.pad_token = pad_token

        def encode(self, smiles):
            tokenized = smi_tokenizer(smiles)
            start_idx = max(0, len(tokenized) - self.sequence_length)
            truncated = smiles[start_idx:]
            padded = (self.sequence_length - len(smiles)) * [self.pad_token] + [self.mapping[element] for element in
                                                                                truncated]
            return padded

        def decode(self, tokenized):
            return ''.join([self.inverse[token] for token in tokenized])

    def get_mappings(df, use_padding=True):
        unique_chars = set(df['SMILES'].str.cat())
        unique_elements = set(df['SMILES'].apply(smi_tokenizer).explode().tolist())

        char_mapping = {char: i + int(use_padding) for i, char in enumerate(sorted(list(unique_chars)))}
        element_mapping = {element: i + int(use_padding) for i, element in enumerate(sorted(list(unique_elements)))}

        return char_mapping, element_mapping

    database_df = load_data(data_path)
    char_mapping, element_mapping = get_mappings(database_df)

    tokenizer = ElementTokenizer(element_mapping)

    return tokenizer


def predict_next_elements(smiles, tokenizer, model, n=10):
    # Convert the SMILES string to a list of integers
    encoded = tokenizer.encode(smiles)
    input_ids = torch.tensor(encoded).unsqueeze(0)

    # Predict the next element
    prediction = model(input_ids)

    # Get the indices of the top n predicted elements
    top_n = torch.topk(prediction[0], n)[1][0].tolist()

    # Get confidence scores for the top n predicted elements
    confidence = torch.softmax(prediction[0], dim=1)[0][top_n].tolist()

    # Return the predicted elements
    return [tokenizer.decode([i]) for i in top_n], confidence


def complete_until_valid(uncomplete_smiles, model, tokenizer, branch_factor, max_depth, current_depth=0):
    """
    Complete a SMILES string until it is valid
    """
    current_depth += 1
    # Predict the next element
    next_elements, confidence = predict_next_elements(uncomplete_smiles, tokenizer, model, n=branch_factor)

    # Append the predicted elements to the SMILES string
    for element in next_elements:
        complete_smiles = uncomplete_smiles + element

        # If the SMILES string is valid, return it
        if is_valid(complete_smiles):
            return complete_smiles

    # If the SMILES string is not valid, and we have reached the maximum depth, return None
    if current_depth == max_depth:
        return None
    else:
        return complete_until_valid(uncomplete_smiles + next_elements[0], model, tokenizer, current_depth=current_depth,
                                    branch_factor=branch_factor, max_depth=max_depth)


def get_model_autocompletions(SMILES, n_max_suggestions, model, tokenizer, max_depth=10):
    next_element_suggestions, confidence = predict_next_elements(SMILES, tokenizer, model, n=n_max_suggestions)
    completions = [complete_until_valid(SMILES + element, model, tokenizer, branch_factor=10, max_depth=max_depth) for
                   element in next_element_suggestions]
    completions = [smiles for smiles in completions if smiles is not None]
    # Remove /, //, \ and \\ from the SMILES strings
    completions = [smiles.replace('/', '').replace('\\', '') for smiles in completions]
    return completions