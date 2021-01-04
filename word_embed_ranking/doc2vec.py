import pickle
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from .utils import read_word_vector_file, get_default_word_vector, get_word_vector_size


def load_pickle(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data


class DocVectorizer:
    def __init__(self, word_embed_file, tfidf_file):
        self.embed = read_word_vector_file(word_embed_file)
        self.tf_idf_model = load_pickle(tfidf_file)
        self.tf_idf_features = self.tf_idf_model.get_feature_names()
        self.word_vector_size = get_word_vector_size(self.embed)

    def _doc_to_word_vectors(self, doc):
        tokens = doc.split()
        word_vectors = [self.embed.get(token, get_default_word_vector(self.word_vector_size))
                        for token in tokens]
        return word_vectors, tokens

    def _get_doc_word_weights(self, doc_vector):
        nonzero_idx = doc_vector.nonzero()[1]

        nonzero_words = [self.tf_idf_features[word_idx] for word_idx in nonzero_idx]
        nonzero_elems = doc_vector[0, nonzero_idx].toarray()[0, :]

        word_weights = dict(zip(nonzero_words, nonzero_elems))
        return word_weights

    def doc_to_vector(self, doc, tf_idf_vector):
        word_vectors, tokens = self._doc_to_word_vectors(doc)
        word_weight_dict = self._get_doc_word_weights(tf_idf_vector)

        word_weights = [word_weight_dict.get(token, 0) for token in tokens]

        word_vectors += [get_default_word_vector(self.word_vector_size)]
        word_weights += [1]

        doc_vector = np.average(word_vectors, axis=0, weights=word_weights)
        return doc_vector

    def transform(self, docs, tf_idf_vectors):
        tfidf_list = [tf_idf_vectors[i] for i in range(tf_idf_vectors.shape[0])]
        docs_with_vectors = zip(docs, tfidf_list)
        doc_vectors = [self.doc_to_vector(*value) for value in tqdm(docs_with_vectors, total=len(docs))]
        doc_vectors = np.array(doc_vectors)
        return doc_vectors


def convert_docs_to_vecs(doc2vec):
    tf_idf_vectors = sparse.load_npz('model/vectors.npz')
    data = pd.read_csv('corpus_cleaned_sample.csv')
    data = data.loc[~data['text'].isna()]
    texts = data['text'].values

    vectors = doc2vec.transform(texts, tf_idf_vectors)
    np.save('model/doc_vectors', vectors)


def convert_queries_to_vecs(doc2vec):
    data = pd.read_csv('data/queries_cleaned_sample.csv')
    print('Loaded data')
    # data = data.loc[~data['text'].isna()]
    texts = data['query'].values

    tf_idf_vectors = doc2vec.tf_idf_model.transform(texts)

    vectors = doc2vec.transform(texts, tf_idf_vectors)
    np.save('model/query_vectors', vectors)


def main():
    doc2vec = DocVectorizer('model/glove.txt', 'model/tfidf_model.pkl')
    # convert_docs_to_vecs(doc2vec)
    convert_queries_to_vecs(doc2vec)


if __name__ == '__main__':
    main()
