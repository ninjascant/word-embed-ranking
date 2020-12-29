import os
import sys
import argparse
import csv
from tqdm.auto import tqdm
import pandas as pd

csv.field_size_limit(sys.maxsize)

DATA_DIR = './raw_data'
TRAIN_QREL_FILE_PATH = os.path.join(DATA_DIR, 'qrels_train.tsv')
DEV_QREL_FILE_PATH = os.path.join(DATA_DIR, 'qrels_dev.tsv')
CORPUS_FILE_PATH = os.path.join(DATA_DIR, 'docs.tsv')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, help='Dataset type: train or dev', default='train')

    args = parser.parse_args()

    if args.type == 'train':
        qrel_file = TRAIN_QREL_FILE_PATH
    elif args.type == 'dev':
        qrel_file = DEV_QREL_FILE_PATH
    else:
        raise RuntimeError('--type arg has only 2 options: train and dev')

    qrels = pd.read_csv(qrel_file, sep=' ', header=None)
    qrels.columns = ['qid', 'none1', 'doc_id', 'none2']
    qrels = qrels[['qid', 'doc_id']]

    train_doc_ids = set(qrels['doc_id'].values)
    print(f'Extracting {args.type} corpus subset')

    processed_rows = []
    with open(CORPUS_FILE_PATH) as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        for i, row in tqdm(enumerate(tsv_reader)):
            if row[0] not in train_doc_ids:
                continue
            processed_rows.append(row)
    processed_rows = pd.DataFrame(processed_rows, columns=['doc_id', 'url', 'title', 'text'])
    processed_rows.to_csv(f'data/{args.type}_corpus.csv', index=False)


if __name__ == '__main__':
    main()
