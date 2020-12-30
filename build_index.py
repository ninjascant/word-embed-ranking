from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from word_embed_ranking import build_annoy_index


def get_query_res(idx, query_vectors, data, index, n_res):
    res = index.get_nns_by_vector(query_vectors[idx], n_res)
    res = list(data.iloc[res]['doc_id'].values)
    true_value = data.iloc[idx]['doc_id']
    if true_value in res:
        reciprocal_rank = 1 / (res.index(true_value) + 1)
    else:
        reciprocal_rank = 0
    return reciprocal_rank


def main():
    doc_vectors = np.load('model/vectors.npy')
    query_vectors = np.load('model/query_vectors.npy')
    data = pd.read_csv('corpus_cleaned_sample.csv')
    data = data.loc[~data['text'].isna()]

    index = build_annoy_index(doc_vectors)

    ranks = [get_query_res(i, query_vectors, data, index, 20)
             for i in tqdm(range(doc_vectors.shape[0]))]
    mrr = np.mean(ranks)
    print(mrr)


if __name__ == '__main__':
    main()






