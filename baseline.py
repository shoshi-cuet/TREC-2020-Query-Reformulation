from elasticsearch import Elasticsearch
from typing import List,Dict,Tuple
import json
from index import INDEX_NAME,index_documents
from evaluation import score_table_by_turn,QRELS
from trectools import TrecRun, TrecEval
import os

def get_utterances(filepath:str = "data/2020_manual_evaluation_topics_v1.0.json")->Dict:

    """
    Function to retrive the differant utterances from the evaluation file from github reposetory. 

    returns dict with the raw utterancec, dict with the manual uterances, dict with the trec auto rewritten utterances

    """
    raw = {}
    manual = {}
    trec_auto = {}
    with open(filepath, 'r') as f:
        file = json.load(f)
    for i in file:
        for passage in i['turn']:
            raw.setdefault(i['number'], []).append(passage['raw_utterance'])
            manual.setdefault(i['number'], []).append(passage['manual_rewritten_utterance'])
            trec_auto.setdefault(i['number'], []).append(passage['automatic_rewritten_utterance'])
    return raw,manual,trec_auto


def calc_bm25(es : Elasticsearch,query:str ,k:int)->List[Tuple]:
    """
    Retrives documents based on BM25 

    returns list of tuples with doc ID and bm25 score
    """
    query_body = {
            "query": {
                    "match": {
                            "passage": query
                            }
                    }
                }
    res = es.search(index=INDEX_NAME, body=query_body, _source=False,size = k)['hits']['hits']
    return [(doc['_id'], doc['_score']) for doc in res]


def retrive_with_bm25(es:Elasticsearch,query:Dict, size:int)->Dict:
    """
    Retrives the top size documents based on BM25

    If writetofile = True, the results is writen to a file, this is for evaluation. 

    es : Elasticsearc instance
    query: dict
    size: number of documents to retrive 
    run_name: name of file for this run

    returns dict 
    """
    
    search={}
    for context_number, querys in query.items():
        for count,i in enumerate(querys):
            search.setdefault(context_number,[]).append({count:calc_bm25(es=es,query=i,k=size)})
    return search

def write_to_file(run_name:str, retrived:Dict) -> None:
    """
    Function to write the retrival scores to a file for easy access when evaluating
    """
    outfile = open(f'{run_name}','w')
    for topic_number,documents in retrived.items():
        for doc in documents:
            for turn,conversation in doc.items():
                for rank,passage in enumerate(conversation):
                    outfile.writelines(f'{topic_number}_{turn+1} q0 {passage[0]} {rank}  {passage[1]} {run_name}\n')


def create_baseline(es:Elasticsearch,index:str, overwrite:bool = False)->None:

    """
    Function to create a baseline

    The function creates an index of the MS marco and the Trec car documents
    When the index i created it retrives the top k = 1000 documents for each query in the 
    data/2020_manual_evaluation_topics_v1.0.json file.

    writes the results to a file and saves the file in the basline/ 
    thise files are used in the evaluation. 

        index: name of the index
        overwrite: overwrites if the file exist 

    """

    #### Creates index if index does not exist
    if not es.indices.exists(index = INDEX_NAME) or es.count(index=INDEX_NAME)['count']<38000000:
        print('Creating index')
        index_documents(es=es,index=INDEX_NAME)
        print('documents are indexed')

    files = ['baseline_results/raw','baseline_results/manual','baseline_results/trec_auto']
    if not os.path.exists(files[0]) and not os.path.exists(files[1]) and not os.path.exists(files[2]) and not overwrite:
        raw,manually,trec = get_utterances()

        ### Retrives the top k = 1000 documents 
        base_line_retried  = retrive_with_bm25(es=es,query=raw,size=1000) 
        manually_rewritten = retrive_with_bm25(es=es,query=manually,size=1000)
        trec_rewritten     = retrive_with_bm25(es=es,query=trec,size=1000)

        ## Writing the results to a file 
        write_to_file(run_name='baseline_results/raw',retrived=base_line_retried)
        write_to_file(run_name='baseline_results/manual',retrived=manually_rewritten)
        write_to_file(run_name='baseline_results/trec_auto',retrived=trec_rewritten)

    print('Baseline files created')

def main():

    es = Elasticsearch()
    es.info()


    create_baseline(es = es,index=INDEX_NAME)
    qrels = QRELS

    ##### Loads the run file into a TrecRun object 
    raw_run       = TrecRun('baseline_results/raw')
    manual_run    = TrecRun('baseline_results/manual')
    trec_auto_run = TrecRun('baseline_results/trec_auto')
    runs = [raw_run,manual_run,trec_auto_run]
    runs_name=['raw_run','manual_run','trec_auto_run']

    ####Evaluation 
    
    ##NDCG
    ndcg1000_overall = [TrecEval(run, qrels).get_ndcg(depth=1000) for run in runs]
    ndcg3_overall = [TrecEval(run, qrels).get_ndcg(depth=3) for run in runs]
    
    ##MAP
    map1000_overall = [TrecEval(run, qrels).get_map(depth=1000) for run in runs]
    
    ##Precition
    precition1000_overall = [TrecEval(run, qrels).get_precision(depth=1000) for run in runs]
    precition3_overall = [TrecEval(run, qrels).get_precision(depth=3) for run in runs]

    #### Write evaluation scores per turn to .csv
    ndcg1000_per_q = [TrecEval(run, qrels).get_ndcg(depth=1000,per_query=True) for run in runs]
    ndcg3_per_q = [TrecEval(run, qrels).get_ndcg(depth=3,per_query=True) for run in runs]
    precition1000_per_q = [TrecEval(run, qrels).get_precision(depth=1000,per_query=True) for run in runs]
    precition3_per_q = [TrecEval(run, qrels).get_precision(depth=3,per_query=True) for run in runs]
    for i in range(len(runs)):
        score_table_by_turn(ndcg1000_per_q[i]).to_csv(f'baseline_results/ndcg1000_per_q{runs_name[i]}.csv')
        score_table_by_turn(ndcg3_per_q[i]).to_csv(f'baseline_results/ndcg3_per_q{runs_name[i]}.csv')
        score_table_by_turn(precition1000_per_q[i]).to_csv(f'baseline_results/precition1000_per_q{runs_name[i]}.csv')
        score_table_by_turn(precition3_per_q[i]).to_csv(f'baseline_results/precition3_per_q{runs_name[i]}.csv')

#### Printing the overall score
    for i in range(len(map1000_overall)):
        print(f'{runs_name[i]} ndcg@1000: {ndcg1000_overall[i]}')
        print(f'{runs_name[i]} ndcg@3: {ndcg3_overall[i]}')
        print(f'{runs_name[i]} map@1000: {map1000_overall[i]}')
        print(f'{runs_name[i]} precition@1000: {precition1000_overall[i]}')
        print(f'{runs_name[i]} precition@3: {precition3_overall[i]}')

if __name__=='__main__':
    main()