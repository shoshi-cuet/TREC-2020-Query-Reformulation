# Dat640

This is the project in DAT640 at the uneversity of stavanger.

The main focus of this project was to automaticly reformulate the querys to keep the context of a conversation as it evolvs. The result of this reformulating is shown in results/utterances.json.

Project files: 
  - index.py 
  - baseline.py
  - main.py
  - evaluation.py
  - rewrite.py 
  - requirements.txt 
  - data/
    -  2020_manual_evaluation_topics_v1.0.json
    -  collection.tsv
    -  dedup.articles-paragraphs.cbor
    -  qrels
  - baseline_results/
    - manual      score file
    - raw         score file
    - trec_auto   score file
  - results/
    - g5_run      score file
    - utterances  all utterances including the groups reformuation of the raw utterances
    - different csv files containing different metric scores 
     
### index.py

The index file contains the information on the elasticsearch index. The index name and index settings. The function for indexing the MS marco and Trac car files are located in the index.py 

### The baseline.py
  The baseline.py contains the functions needed for creating the baseline for the project
  
### main.py
  Contains the functions for reproduce the results represented in the report
  
### evaluation.py
  contains a function for making a pandas dataframe of the result of a metric 
  
### rewrite.py 
  contains the class that is used to reformulationg the queries. This class usses a pretraind t5 model from huggingface. https://huggingface.co/castorini/t5-base-canard
    
### Steps to reproduce the result presented in the report
  1. Install the requirements in the requirment.txt 
  2. Run the main.py file'
