import argparse
import pandas as pd
import fasttext


def prepare_corpus_file(sample_size):
    if sample_size is None:
        texts = pd.read_csv('data/corpus_cleaned_train.csv', usecols=['text'])
    else:
        texts = pd.read_csv('data/corpus_cleaned_train.csv', usecols=['text'], nrows=sample_size)
    texts.loc[texts['text'].isna(), 'text'] = ''
    texts = texts['text'].values
    texts = '\n'.join(texts)

    with open('data/train_texts.txt', 'w') as outfile:
        outfile.writelines(texts)


def train_model():
    model = fasttext.train_unsupervised('data/train_texts.txt', dim=200, epoch=1, thread=4)
    nn = model.get_nearest_neighbors('nit')
    print(nn)
    model.save_model("fasttext.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--sample-size', type=int, default=None)
    args = parser.parse_args()

    if args.task == 'prepare_corpus':
        prepare_corpus_file(args.sample_size)
    elif args.task == 'train_model':
        train_model()


if __name__ == '__main__':
    main()
