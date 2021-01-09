import logging
import pickle
import pandas as pd
from .preprocess_corpus import TextCleaner, Lemmatizer, StemmerLemmatizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(f_format)
logger.addHandler(c_handler)


def preprocess_for_tfidf(corpus_file, outfile, sample_size, use_stemming, max_len):
    if sample_size:
        data = pd.read_csv(corpus_file, nrows=sample_size)
    else:
        data = pd.read_csv(corpus_file)
    data.loc[data['text'].isna(), 'text'] = ''

    text_cleaner = TextCleaner()
    if use_stemming:
        lemmatizer = StemmerLemmatizer(max_len=max_len)
    else:
        lemmatizer = Lemmatizer(max_len=max_len, remove_pronouns=True)

    texts = data['text'].values
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
