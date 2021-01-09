import logging
import pickle
import numpy as np
import pandas as pd
from .preprocess_corpus import TextCleaner, Lemmatizer, StemmerLemmatizer
from .doc_vectorizer import WordCentroidVectorizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(f_format)
logger.addHandler(c_handler)


def preprocess_for_tfidf(corpus_file, outfile, sample_size, use_stemming, max_len, text_field):
    if sample_size:
        skip = range(1, 50_001)
        data = pd.read_csv(corpus_file, nrows=sample_size, skiprows=skip)
    else:
        data = pd.read_csv(corpus_file)
    data.loc[data[text_field].isna(), text_field] = ''

    text_cleaner = TextCleaner()
    if use_stemming:
        lemmatizer = StemmerLemmatizer(max_len=max_len)
    else:
        lemmatizer = Lemmatizer(max_len=max_len, remove_pronouns=True)

    texts = data[text_field].values
    logger.info('Preprocessing texts')
    cleaned_texts = text_cleaner.transform(texts)
    logger.info('Lemmatizing texts')
    lemmatized_texts = lemmatizer.transform(cleaned_texts)

    # with open(outfile, 'wb') as outfile:
    #     pickle.dump(lemmatized_texts, outfile)

    lemmatized_texts = [{'original': text.original_text, 'preprocessed': text.preprocessed_text}
                        for text in lemmatized_texts]
    lemmatized_texts = pd.DataFrame(lemmatized_texts)
    lemmatized_texts.to_csv(outfile, index=False)


def vectorize_docs(corpus_file, outfile):
    data = pd.read_csv(corpus_file)
    data.loc[data['original'].isna(), 'original'] = ''
    texts = data['original'].values

    vectorizer = WordCentroidVectorizer(
        embed_file_path='model/glove.txt',
        n_top_terms=20,
        tf_idf_file='tf_idf_model.pkl'
    )
    vectorizer.fit()
    doc_vectors = vectorizer.transform(texts)
    np.save(outfile, doc_vectors)
