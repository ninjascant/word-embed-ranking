import time
import pickle
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.parsing.preprocessing import strip_punctuation, strip_numeric, remove_stopwords, strip_short, \
    stem_text

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(f_format)
logger.addHandler(c_handler)

GLOVE_PIPELINE_FILTERS = [strip_punctuation, strip_numeric, remove_stopwords, strip_short]
TFIDF_PIPELINE_FILTERS = [stem_text]


def read_word_vector_file(file_path):
    embedding_dict = {}
    with open(file_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], 'float32')
            embedding_dict[word] = vector
    return embedding_dict


def preprocess_text(doc, filters):
    return gensim.parsing.preprocessing.preprocess_string(doc, filters)


def train_tfidf(train_texts, max_features, ngrams=(1, 1), filename='tfidf_model.pkl', return_vectors=True):
    tfidf_model = TfidfVectorizer(
        stop_words='english',
        ngram_range=ngrams,
        max_features=max_features
    )
    if return_vectors:
        start_time = time.time()
        tfidf_vectors = tfidf_model.fit_transform(train_texts)
        elapsed = '%.2f' % (time.time() - start_time)
        logger.info(f'Trained tf-idf model. Elapsed: {elapsed}s')
        pickle.dump(tfidf_model, open(filename, 'wb'))
        return tfidf_model, tfidf_vectors
    else:
        start_time = time.time()
        tfidf_model.fit(train_texts)
        elapsed = '%.2f' % (time.time() - start_time)
        logger.info(f'Trained tf-idf model. Elapsed: {elapsed}s')
        pickle.dump(tfidf_model, open(filename, 'wb'))
        return tfidf_model


def get_tfidf_word_weights(vector, tfidf_features):
    non_zero_elems = vector.nonzero()[1]
    word_weights = {tfidf_features[idx]: vector[0, idx] for idx in non_zero_elems}
    return word_weights
