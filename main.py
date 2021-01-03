import argparse
import pickle
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from word_embed_ranking import WordCentroidVectorizer, build_annoy_index

tqdm.pandas()


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def split_doc(row):
    doc_splited = row['text_truncated'].split()
    doc_splited = chunks(doc_splited, 20)
    doc_splited = [(' '.join(text_chunk), row['doc_id'], row['query']) for text_chunk in doc_splited]
    return doc_splited


def split_docs():
    data = pd.read_csv('data/val_data.csv')
    splitted_docs = data.progress_apply(split_doc, axis=1)
    splitted_docs = [item for items in splitted_docs for item in items]
    splitted_docs = pd.DataFrame(splitted_docs, columns=['text_truncated', 'doc_id', 'query'])
    splitted_docs.to_csv('data/splited_docs.csv', index=False)


def get_query_res(idx, query_vectors, data, index, n_res):
    res = index.get_nns_by_vector(query_vectors[idx], n_res)
    res = list(data.iloc[res]['doc_id'].values)
    true_value = data.iloc[idx]['doc_id']
    if true_value in res:
        reciprocal_rank = 1 / (res.index(true_value) + 1)
    else:
        reciprocal_rank = 0
    return reciprocal_rank, res


def compute_mrr():
    doc_vectors = np.load('model/doc_vectors.npy')
    query_vectors = np.load('model/query_vectors.npy')
    data = pd.read_csv('data/val_data.csv')
    data = data.loc[~data['text_truncated'].isna()]

    index = build_annoy_index(doc_vectors, num_trees=300)

    sample_size = 10_000 # doc_vectors.shape[0]

    ranking_res = [get_query_res(i, query_vectors, data, index, 20)
                   for i in tqdm(range(sample_size))]
    ranks = [item[0] for item in ranking_res]

    mrr = np.mean(ranks)
    mrr = '%.5f' % mrr
    print(f'MRR@20: {mrr}')

    data = data.iloc[:sample_size]
    data['rank'] = ranks
    print('%.2f' % (data.loc[data['rank'] == 0].shape[0] * 100 / sample_size))
    data[['doc_id', 'qid', 'query', 'rank']].to_csv('ranked_queries.csv', index=False)


def construct_search_embeddings(tf_idf_model_file):
    data = pd.read_csv('data/val_data.csv', usecols=['text_truncated', 'query'])
    data = data.loc[~data['text_truncated'].isna()]
    corpus_texts = data['text_truncated'].values
    query_texts = data['query'].values

    doc2vec = WordCentroidVectorizer('model/glove.txt', n_top_terms=None, tf_idf_file=tf_idf_model_file)

    doc2vec.fit(corpus_texts)
    with open('model/doc2vec.pkl', 'wb') as outfile:
        pickle.dump(doc2vec, outfile)

    doc_vectors = doc2vec.transform(corpus_texts)
    query_vectors = doc2vec.transform(query_texts)

    np.save('model/doc_vectors', doc_vectors)
    np.save('model/query_vectors', query_vectors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str)
    parser.add_argument('--tf-idf-model', type=str, default=None)
    args = parser.parse_args()
    task = args.task

    if task == 'construct_search_embeddings':
        construct_search_embeddings(args.tf_idf_model)
    elif task == 'compute_mrr':
        compute_mrr()
    elif task == 'split_docs':
        split_docs()


if __name__ == '__main__':
    main()
