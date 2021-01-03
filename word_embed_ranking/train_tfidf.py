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


def main():
    data = pd.read_csv('data/splited_docs.csv')
    data = data.loc[~data['text_truncated'].isna()]
    train_data = data.iloc[:300_000]
    val_data = data.iloc[300_000:]
    texts = train_data['text_truncated'].values

    train_tfidf(texts, 200_000)
    val_data.to_csv('val_data.csv', index=False)
    train_data.to_csv('train_data.csv', index=False)


if __name__ == '__main__':
    main()
