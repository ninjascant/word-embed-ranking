import logging
import pickle
from tqdm.auto import tqdm
import numpy as np
from sklearn.preprocessing import Normalizer
import scipy.sparse as sparse
from .utils import read_word_vector_file, get_word_vector_size, get_default_word_vector, EN_STOP_WORDS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(f_format)
logger.addHandler(c_handler)


def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


class WordCentroidVectorizer:
    def __init__(self, embed_file_path, n_top_terms=20, tf_idf_file=None):
        self.n_top_terms = n_top_terms

        word_embed_dict, word_embed_vocab, word_embed_size = self._prepare_word_embeddings(embed_file_path)
        self.word_embed_dict = word_embed_dict
        self.word_embed_size = word_embed_size

        self.tf_idf_model = load_pickle_file(tf_idf_file)

        self.normalizer = Normalizer(norm='l2')
        self.vocab = None

    @staticmethod
    def _prepare_word_embeddings(embed_file_path):
        logger.info('Loading word vector file...')
        word_embed_dict = read_word_vector_file(embed_file_path)
        logger.info('Loaded word vector file')

        word_embed_vocab = word_embed_dict.keys()
        word_embed_vocab = [word for word in word_embed_vocab if word not in EN_STOP_WORDS]

        word_embed_size = get_word_vector_size(word_embed_dict)

        return word_embed_dict, word_embed_vocab, word_embed_size

    def _construct_word_embed_matrix(self):
        vocab = self.tf_idf_model.get_feature_names()
        vocab_size = len(vocab)
        word_embed_matrix = np.zeros((vocab_size, self.word_embed_size))

        for i, word in enumerate(vocab):
            word_embed_matrix[i] = self.word_embed_dict.get(word, get_default_word_vector(self.word_embed_size))

        self.word_embed_matrix = word_embed_matrix

    def fit(self):
        self.vocab = self.tf_idf_model.get_feature_names()

        logger.info('Start constructing word embedding matrix')
        self._construct_word_embed_matrix()
        logger.info('Constructed word embedding matrix')

    def _get_doc_word_weights(self, doc_vector):
        nonzero_idx = doc_vector.nonzero()[1]
        nonzero_elems = doc_vector[0, nonzero_idx].toarray()[0, :]

        word_weights = zip(nonzero_idx, nonzero_elems)
        word_weights = sorted(word_weights, key=lambda x: x[1], reverse=True)[self.n_top_terms:]
        elems_to_zero = [weight[0] for weight in word_weights]
        doc_vector[0, elems_to_zero] = 0
        return doc_vector

    def _get_top_terms_for_texts(self, vectors):
        top_term_vectors = [self._get_doc_word_weights(vectors[i]) for i in tqdm(range(vectors.shape[0]))]
        top_term_vectors = sparse.vstack(top_term_vectors)
        return top_term_vectors

    def transform(self, texts):
        term_occur_matrix = self.tf_idf_model.transform(texts)
        if self.n_top_terms is not None:
            term_occur_matrix = self._get_top_terms_for_texts(term_occur_matrix)
        word_centroid_matrix = term_occur_matrix.dot(self.word_embed_matrix)
        word_centroid_matrix = self.normalizer.transform(word_centroid_matrix)

        return word_centroid_matrix

    def fit_transform(self, texts):
        self.fit()
        return self.transform(texts)
