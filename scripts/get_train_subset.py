import os
import argparse
import pandas as pd

DATA_DIR = './data'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=1000)

    args = parser.parse_args()

    dataset = pd.read_csv(os.path.join(DATA_DIR, 'corpus_truncated_train.csv'), nrows=args.sample_size)
    dataset.to_csv(os.path.join(DATA_DIR, 'corpus_truncated_train_sample.csv'), index=False)


if __name__ == '__main__':
    main()
