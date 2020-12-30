import time
import logging
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(f_format)
logger.addHandler(c_handler)


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


def main():
    data = pd.read_csv('corpus_cleaned_sample.csv')
    texts = data['text'].values

    model, vectors = train_tfidf(texts, 100_000, return_vectors=True)
    sparse.save_npz('vectors', vectors)


if __name__ == '__main__':
    main()

