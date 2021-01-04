import unittest
import numpy as np
from word_embed_ranking import DocVectorizer


class DocVectorizerTest(unittest.TestCase):
    @staticmethod
    def test_fit_call():
        test_vector_file = './tests/glove_sample.txt'
        corpus_file = '../msmarco_data/corpus_splitted.csv'
        vectorizer = DocVectorizer(test_vector_file, corpus_file, 1000, corpus_sample_size=100)
        vectorizer.fit()
        assert vectorizer.doc_embeddings.shape == (100, 100)
        assert vectorizer.doc_embeddings.dtype == np.float32

    @staticmethod
    def test_get_query_embedding_with_valid_query():
        test_vector_file = './tests/glove_sample.txt'
        corpus_file = '../msmarco_data/corpus_splitted.csv'
        vectorizer = DocVectorizer(test_vector_file, corpus_file, 1000, corpus_sample_size=100)
        vectorizer.fit()

        query = 'zinc welding job'
        query_vector = vectorizer.get_query_embedding(query)

        assert not np.array_equal(query_vector, np.zeros(100, dtype=np.float32))
        assert query_vector.shape == (100,)

    @staticmethod
    def test_get_query_embedding_with_invalid_query():
        test_vector_file = './tests/glove_sample.txt'
        corpus_file = '../msmarco_data/corpus_splitted.csv'
        vectorizer = DocVectorizer(test_vector_file, corpus_file, 1000, corpus_sample_size=100)
        vectorizer.fit()

        query = 'diet shakes for diabetics'
        query_vector = vectorizer.get_query_embedding(query)

        assert np.array_equal(query_vector, np.zeros(100, dtype=np.float32))
