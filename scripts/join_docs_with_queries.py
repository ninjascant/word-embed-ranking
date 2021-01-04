import os
import pandas as pd

RAW_DATA_DIR = './raw_data'
CORPUS_DATA_DIR = './data'

TRAIN_QREL_FILE_PATH = os.path.join(RAW_DATA_DIR, 'qrels_train.tsv')
TRAIN_QUERIES_FILE_PATH = os.path.join(RAW_DATA_DIR, 'queries_train.tsv')
TRAIN_CORPUS_FILE = os.path.join(CORPUS_DATA_DIR, 'train_corpus.csv')

DEV_QREL_FILE_PATH = os.path.join(RAW_DATA_DIR, 'qrels_dev.tsv')
DEV_QUERIES_FILE_PATH = os.path.join(RAW_DATA_DIR, 'queries_dev.tsv')
DEV_CORPUS_FILE = os.path.join(CORPUS_DATA_DIR, 'dev_corpus.csv')


def read_files(query_file, qrel_file, corpus_file):
    queries = pd.read_csv(query_file, sep='\t', header=None)
    qrels = pd.read_csv(qrel_file, sep=' ', header=None)

    queries.columns = ['qid', 'query']
    qrels.columns = ['qid', 'none1', 'doc_id', 'none2']

    qrels = qrels[['qid', 'doc_id']]
    corpus = pd.read_csv(corpus_file)
    return queries, qrels, corpus


def join_docs_with_queries(queries, qrels, corpus, type):
    if 'qid' not in corpus.columns:
        queries = queries.merge(qrels, on='qid', how='outer')
        corpus = corpus.merge(queries, on='doc_id', how='outer')
        corpus.to_csv(os.path.join(CORPUS_DATA_DIR, f'corpus_with_queries_{type}.csv'), index=False)
    else:
        print(f'{type} already joined')


def main():
    train_queries, train_qrels, train_corpus = read_files(
        TRAIN_QUERIES_FILE_PATH,
        TRAIN_QREL_FILE_PATH,
        TRAIN_CORPUS_FILE
    )
    dev_queries, dev_qrels, dev_corpus = read_files(
        DEV_QUERIES_FILE_PATH,
        DEV_QREL_FILE_PATH,
        DEV_CORPUS_FILE
    )

    join_docs_with_queries(train_queries, train_qrels, train_corpus, 'train')
    join_docs_with_queries(dev_queries, dev_qrels, dev_corpus, 'dev')


if __name__ == '__main__':
    main()
