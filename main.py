import argparse
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from word_embed_ranking import build_annoy_index
from word_embed_ranking import preprocess_for_tfidf, vectorize_docs


def get_query_res(idx, query_vectors, data, index, n_res):
    res = index.get_nns_by_vector(query_vectors[idx], n_res)
    res = list(data.iloc[res]['doc_id'].values)
    true_value = data.iloc[idx]['doc_id']
    if true_value in res:
        reciprocal_rank = 1 / (res.index(true_value) + 1)
    else:
        reciprocal_rank = 0
    return reciprocal_rank, res


def compute_mrr(corpus_file, outfile):
    doc_vectors = np.load('model/doc_vectors.npy')
    query_vectors = np.load('model/query_vectors.npy')
    data = pd.read_csv(
        corpus_file,
        # skiprows=range(1, 50_001),
        nrows=10_000
    )

    index = build_annoy_index(doc_vectors, num_trees=300)

    sample_size = doc_vectors.shape[0]

    ranking_res = [get_query_res(i, query_vectors, data, index, 100)
                   for i in tqdm(range(sample_size))]
    ranks = [item[0] for item in ranking_res]

    mrr = np.mean(ranks)
    mrr = '%.5f' % mrr
    print(f'MRR@20: {mrr}')

    data = data.iloc[:sample_size]
    data['rank'] = ranks
    print('%.2f' % (data.loc[data['rank'] == 0].shape[0] * 100 / sample_size))
    data[['doc_id', 'qid', 'query', 'rank']].to_csv(outfile, index=False)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str)
    parser.add_argument('--corpus-file', type=str)
    parser.add_argument('--outfile', type=str)
    parser.add_argument('--sample-size', type=int)
    parser.add_argument('--use-stemming', type=str2bool, default=False)
    parser.add_argument('--max-len', type=int, default=None)
    parser.add_argument('--text-field', type=str)
    parser.add_argument('--tf-idf-model', type=str, default=None)
    args = parser.parse_args()

    task = args.task

    if task == 'preprocess':
        preprocess_for_tfidf(args.corpus_file, args.outfile, args.sample_size, args.use_stemming, args.text_field)
    elif task == 'vectorize':
        vectorize_docs(args.corpus_file, args.outfile)
    elif task == 'compute_mrr':
        compute_mrr(args.corpus_file, args.outfile)


if __name__ == '__main__':
    main()
