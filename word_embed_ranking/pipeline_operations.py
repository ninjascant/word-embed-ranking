import logging
import pickle
import numpy as np
import pandas as pd
from .preprocess_corpus import TextCleaner, Lemmatizer
from .doc_vectorizer import WordCentroidVectorizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(f_format)
logger.addHandler(c_handler)


def preprocess_for_tfidf(corpus_file, outfile, sample_size, use_stemming, text_field):
    if sample_size:
        data = pd.read_csv(corpus_file, nrows=sample_size)
    else:
        data = pd.read_csv(corpus_file)
    data.loc[data[text_field].isna(), text_field] = ''

    text_cleaner = TextCleaner()
    lemmatizer = Lemmatizer(n_jobs=8, remove_pronouns=False)

    texts = data[text_field].values
    cleaned_texts = text_cleaner.transform(texts)
    lemmatized_texts = lemmatizer.transform(cleaned_texts)
    lemmatized_texts = pd.DataFrame(lemmatized_texts, columns=['text'])
    lemmatized_texts.to_csv(outfile, index=False)


def vectorize_docs(corpus_file, outfile):
    data = pd.read_csv(corpus_file)
    data.loc[data['preprocessed'].isna(), 'preprocessed'] = ''
    texts = data['preprocessed'].values

    vectorizer = WordCentroidVectorizer(
        embed_file_path='model/glove.txt',
        n_top_terms=20,
        tf_idf_file='model/tf_idf_model.pkl'
    )
    vectorizer.fit()
    doc_vectors = vectorizer.transform(texts)
    np.save(outfile, doc_vectors)
