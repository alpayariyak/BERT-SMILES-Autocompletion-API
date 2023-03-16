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

The BERT model was fine-tuned using a dataset of valid SMILES strings. Additional generated SMILES strings were also generated using the RDKit library.
