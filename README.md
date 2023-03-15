<h1 align="center">
  <b>BERT SMILES Autocompletion + API</b><br> 

</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.8-ff69b4.svg" /></a>
       <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.12-2BAF2B.svg" /></a>
        <a href="https://flask.palletsprojects.com/en/2.0.x/">
        <img src="https://img.shields.io/badge/Flask-2.2-000000.svg" /></a>
        <a href="https://huggingface.co/docs/transformers/index">
        <img src="https://img.shields.io/badge/Transformers-4.19.4-FF6F00.svg" /></a>
        <a href="https://www.rdkit.org/">
        <img src="https://img.shields.io/badge/RDKit-2022.9.3-00CC00.svg" /></a>


</p>

BERT SMILES Autocompletion + API is a project fine-tuning and deploying a BERT model to predict the next element and character in a SMILES (Simplified Molecular Input Line Entry System) string. The API allows users to autocomplete SMILES strings with high accuracy, making it easier to access molecules without using drawing software.

![demo.png](github_images%2Fdemo.png)
<p style="text-align: center;">Web App Demo of SMILES Autocompletion API</p>

# Table of Contents
  - [Fine-Tuned BERT Model for SMILES Autocompletion](#fine-tuned-bert-model-for-smiles-autocompletion)
    - [Algorithm for SMILES Autocompletion with BERT](#algorithm-for-smiles-autocompletion-with-bert)
    - [Advantages of using Model over Database Search](#advantages-of-using-model-over-database-search)
  - [BERT SMILES Autocompletion API](#bert-smiles-autocompletion-api)
    - [Installation](#installation)
    - [Usage](#usage)
      - [Endpoints](#endpoints)
      - [Query Parameters](#query-parameters)

# Fine-Tuned BERT Model for SMILES Autocompletion

The BERT model was fine-tuned using a dataset of valid SMILES strings. Additional generated SMILES strings were also generated using the RDKit library. The dataset was preprocessed to create masked language model (MLM) training examples, where a portion of the SMILES strings was masked with a special __[MASK]__ token. The objective of the MLM task is to predict the masked tokens based on the context provided by the surrounding unmasked tokens.

During the fine-tuning process, the model learned the syntactic and semantic patterns within the SMILES strings, enabling it to generate chemically valid suggestions for the masked positions.
## Algorithm for SMILES Autocompletion with BERT
![model_explained.png](github_images%2Fmodel_explained.png)
<p style="text-align: center;">Algorithm for SMILES Autocompletion. </p>

## Advantages of using Model over Database Search
- **Expanded Chemical Space**: The model can generate exponentially more valid SMILES strings based on learned patterns, enabling exploration of novel and unexplored chemical structures.
- **Robustness and Flexibility**: The model adapts to different input SMILES strings and generates contextually appropriate suggestions, leading to more accurate and diverse results.
- **Reduced Dependency on Database Size and Quality**: By leveraging the model's learning capabilities, dependency on databases is minimized, making the autocompletion process more efficient and scalable.


# BERT SMILES Autocompletion API
## Installation
To set up and run the BERT SMILES Autocompletion API, follow these steps:

1.  Clone the repository:
    ```
    $ git clone https://github.com/alpayariyak/BERT-SMILES-Autocompletion-API.git 
    $ cd BERT-SMILES-Autocompletion-API
    ```
2. Install the required packages:
    ```
    $ pip install -r requirements.txt
    ```
3. Run the Flask app:
    ```
    $ python autocompletionAPI.py
    ```
The API will be accessible at http://localhost:5000.

## Usage
### Endpoints
>**/autocomplete**: autocompletes a given SMILES string using the fine-tuned BERT model, the database search, or both.

### Query Parameters
>**smiles**: The SMILES string to autocomplete. (required)

> **n_max_suggestions**: The maximum number of suggestions to return (default: 5).

> **use_model**: Set to true to use the BERT model for autocompletion (default: true).

> **use_database**: Set to true to use the database search for autocompletion (default: true).

> **max_search_length**: The maximum depth to search when using the BERT model (default: 10).


