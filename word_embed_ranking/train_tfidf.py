import time
import logging
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(f_format)
logger.addHandler(c_handler)


def train_tfidf(train_texts, max_features, ngrams=(1, 1), outfile='tf_idf_model.pkl'):
    tf_idf_model = TfidfVectorizer(
        stop_words='english',
        ngram_range=ngrams,
        max_features=max_features
    )
    start_time = time.time()
    logger.info('Training tf-idf model...')
    tfidf_vectors = tf_idf_model.fit_transform(train_texts)
    elapsed = '%.2f' % (time.time() - start_time)
    logger.info(f'Trained tf-idf model. Elapsed: {elapsed}s')
    with open(outfile, 'wb') as file:
        pickle.dump(tf_idf_model, file)
    return tf_idf_model, tfidf_vectors


def remove_sent(text):
    return ' '.join(text.split('.'))


def main():
    data = pd.read_csv('data/tfidf_train_texts.csv')
    data.loc[data['original'].isna(), 'original'] = ''
    texts = data['original'].values
    texts = [remove_sent(text) for text in texts]

    train_tfidf(texts, 200_000)


if __name__ == '__main__':
    main()
