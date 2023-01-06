from rewrite import Rewrite
from baseline import get_utterances,retrive_with_bm25,write_to_file
from elasticsearch import Elasticsearch
from trectools import TrecRun, TrecEval
import json
from evaluation import score_table_by_turn,QRELS
from baseline import create_baseline
from index import INDEX_NAME
from typing import Dict

def reformulate()->Dict:
    """
    Function for reformulating the the raw utterances
    returns dict with the reformulated sentences for each turn in each topic. 
    """
    ### retrives the utterances from  2020_manual_evaluation_topics_v1.0.json
    raw_utterances,_,_ = get_utterances() 

    ## Rewrites the raw utterances by the Rewrite class
    rewriter = Rewrite()
    g5_rewrites = {}
    for topic_number, turn in raw_utterances.items():
        rewriter.reset_context()
        for i in turn:
            g5_rewrites.setdefault(topic_number,[]).append(rewriter.rewrite(query=i))

    return g5_rewrites


def create_json_file(rewrites:Dict)->None:
    """
    Function to create a file containing all the different queryes. 

    rewrites: Dict with reformulated utterances 

    creates a json file
    """
    raw_utterances,man_utterances,_trec_utterances = get_utterances()
    tmp = {}
    for (raw_key,raw_value), (manual_key,manual_value),(trec_key,trec_value),(g5_key,g5_value) in zip(raw_utterances.items(), man_utterances.items(),_trec_utterances.items(),rewrites.items()):
        for i,(raw_v,man_v,trec_v,g5_v) in enumerate(zip(raw_value,manual_value,trec_value,g5_value)):
            
            tmp.setdefault(raw_key,[]).append({i:{'raw utterance':raw_v,
                                                  'manual rewritten utterance':man_v,
                                                  'trec auto rewritten utterance':trec_v,
                                                  'g5 rewritten utterance':g5_v}})
        
    with open("results/utterances.json", "w") as outfile:
        json.dump(tmp, outfile,indent=4,sort_keys=True)

def main():

    es = Elasticsearch(timeout = 50)
    es.info()

    create_baseline(es=es,index=INDEX_NAME)
    g5_rewrites = reformulate()

    create_json_file(rewrites=g5_rewrites)


    ##### RETRIVING with BM25#############
    print('Starting to retrive')
    g5_utterances_run = retrive_with_bm25(es=es,query=g5_rewrites,size=1000)
    write_to_file(run_name='results/g5_run',retrived=g5_utterances_run) 


##########EVALUATION########################
    qrels = QRELS

    #Loading the scores into TrecRun Object
    raw_run = TrecRun('baseline_results/raw')
    manual_run = TrecRun('baseline_results/manual')
    trec_auto_run = TrecRun('baseline_results/trec_auto')
    g5_run = TrecRun('results/g5_run')

    runs = [raw_run,manual_run,trec_auto_run,g5_run]
    run_names = ['raw_run','manual_run','trec_auto_run','g5_run']

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
        score_table_by_turn(ndcg1000_per_q[i]).to_csv(f'results/ndcg1000_per_q{run_names[i]}.csv')
        score_table_by_turn(ndcg3_per_q[i]).to_csv(f'results/ndcg3_per_q{run_names[i]}.csv')
        score_table_by_turn(precition1000_per_q[i]).to_csv(f'results/precition1000_per_q{run_names[i]}.csv')
        score_table_by_turn(precition3_per_q[i]).to_csv(f'results/precition3_per_q{run_names[i]}.csv')

#### Printing the overall score
    for i in range(len(run_names)):
        print(f'{run_names[i]} ndcg@1000: {ndcg1000_overall[i]}')
        print(f'{run_names[i]} ndcg@3: {ndcg3_overall[i]}')
        print(f'{run_names[i]} map@1000: {map1000_overall[i]}')
        print(f'{run_names[i]} precition@1000: {precition1000_overall[i]}')
        print(f'{run_names[i]} precition@3: {precition3_overall[i]}')


if __name__=="__main__":
    main()