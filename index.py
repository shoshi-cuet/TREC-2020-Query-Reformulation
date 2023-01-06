from elasticsearch import Elasticsearch, helpers
from trec_car import read_data
import csv

INDEX_NAME = "database"
INDEX_SETTINGS = {
    "settings" : {
        "index" : {
            "number_of_shards" : 1,
            "number_of_replicas" : 1
        },
        "analysis": {
            "analyzer": {
                "my_english_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "stopwords": "_english_",
                    "filter": [
                        "lowercase",
                        "english_stop",
                        "filter_english_minimal"
                    ]                
                }
            },
            "filter" : {
                "filter_english_minimal" : {
                    "type": "stemmer",
                    "name": "minimal_english"
                },
                "english_stop": {
                    "type": "stop",
                    "stopwords": "_english_"
                }
            },
        }
    },
    "mappings": {
        "properties": {
            "passage": {
                "type": "text",
                "analyzer": "my_english_analyzer"
            }
        }
    }
}


def index_documents(es: Elasticsearch, index: str) -> None:

    """
    Function for indexing the documents. 

    es: Elasticsearch instance
    index: index name 
    
    """

    if es.indices.exists(index = INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)

    es.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)

    MS_Marco_path = "data/collection.tsv"
    car_file = "data/dedup.articles-paragraphs.cbor"

    def gen_MSMARCO_data():
        with open(MS_Marco_path) as f:
            print(f'Loading MS Marco ....')
            msmarco = csv.reader(f, delimiter="\t")
            for line in msmarco:
                docid, text = line
                yield {
                    "_index": index,
                    "_id": f"MARCO_{docid}",
                    "passage": text,
                }

    def gen_TREC_data():
        with open(car_file, 'rb') as f:
            print(f'Loading trec_car.cbor......')
            for para in read_data.iter_paragraphs(f):
                yield {
                    "_index": index,
                    "_id": f"CAR_{para.para_id}",
                    "passage": f"CAR_{para.para_id}",
                }

    helpers.bulk(client=es,actions= gen_MSMARCO_data())
    helpers.bulk(client=es,actions= gen_TREC_data())

