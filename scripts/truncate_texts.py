import os
import argparse
from tqdm.auto import tqdm
import pandas as pd
from nltk.tokenize import word_tokenize
try:
    word_tokenize('hello world')
except LookupError:
    import nltk
    nltk.download('punkt')

tqdm.pandas()

DATA_DIR = 'data'
TRAIN_CORPUS_FILE = os.path.join(DATA_DIR, 'corpus_with_queries_train.csv')
DEV_CORPUS_FILE = os.path.join(DATA_DIR, 'corpus_with_queries_dev.csv')


def process_data_chunk(chunk):
    chunk = chunk.loc[~chunk['text'].isna()].reset_index(drop=True)
    chunk['words'] = chunk['text'].progress_apply(lambda x: word_tokenize(x))
    chunk['original_word_count'] = chunk['words'].apply(lambda x: len(x))
    chunk['text_truncated'] = chunk['words'].apply(lambda x: ' '.join(x[:1000]))
    chunk = chunk.drop(columns=['text', 'words'], axis=1)
    return chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, help='Dataset type: train or dev', default='train')
    parser.add_argument('--sample', type=int, help='Sample size', default=None)

    args = parser.parse_args()

    if args.type == 'train':
        file_path = TRAIN_CORPUS_FILE
    elif args.type == 'dev':
        file_path = DEV_CORPUS_FILE
    else:
        raise RuntimeError('--type arg has only 2 options: train and dev')

    if args.sample is not None:
        data_reader = pd.read_csv(file_path, iterator=True, chunksize=50_000, nrows=args.sample)
    else:
        data_reader = pd.read_csv(file_path, iterator=True, chunksize=50_000)
    processed_chunks = []
    for i, chunk in enumerate(data_reader):
        chunk = process_data_chunk(chunk)
        processed_chunks.append(chunk)
    processed_data = pd.concat(processed_chunks)
    processed_data.to_csv(os.path.join(DATA_DIR, f'corpus_truncated_{args.type}.csv'), index=False)


if __name__ == '__main__':
    main()