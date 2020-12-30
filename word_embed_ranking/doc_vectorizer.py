import logging
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from .utils import read_word_vector_file, preprocess_text, get_tfidf_word_weights, train_tfidf, \
    GLOVE_PIPELINE_FILTERS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
c_handler = logging.StreamHandler()
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(f_format)
logger.addHandler(c_handler)

stemmer = SnowballStemmer("english")


def stem_text(text):
    if type(text) == str:
        text = text.split()
    stemmed_text = list(map(stemmer.stem, text))
    return stemmed_text


class DocVectorizer:
    def __init__(self, word_vector_file, corpus_file, tfidf_max_features, corpus_sample_size=None,
                 use_stemming=False):
        self.word_vector_dict = self._load_word_vectors(word_vector_file)
        self.corpus = self._load_corpus(corpus_file, corpus_sample_size)
        self.texts = self.corpus['text'].values
        self.tfidf_max_features = tfidf_max_features
        self.word_vector_size = next(iter(self.word_vector_dict.values())).shape[0]
        self.use_stemming = use_stemming

        self.cleaned_texts_tokenized = None
        self.cleaned_texts = None
        self.tfidf_features = None
        self.doc_embeddings = None

    @staticmethod
    def _load_word_vectors(file_path):
        logger.info('Start loading word vector file...')
        vectors = read_word_vector_file(file_path)
        logger.info('Loaded word vectors')
        return vectors

    @staticmethod
    def _load_corpus(file_path, sample_size=None):
        if sample_size is None:
            corpus = pd.read_csv(file_path)
        else:
            corpus = pd.read_csv(file_path, nrows=sample_size)
        corpus = corpus.loc[~corpus['text'].isna()]
        corpus['text'] = corpus['text'].apply(lambda x: x.lower())
        corpus = corpus.reset_index(drop=True)
        return corpus

    @staticmethod
    def _preprocess_doc(doc, use_stemming):
        cleaned_doc = preprocess_text(doc, GLOVE_PIPELINE_FILTERS)
        if use_stemming:
            stemmed_doc = stem_text(cleaned_doc)
            return stemmed_doc
        else:
            return cleaned_doc

    def _preprocess_texts(self):
        self.cleaned_texts_tokenized = [self._preprocess_doc(word, self.use_stemming) for word in self.texts]
        self.cleaned_texts = [' '.join(text) for text in self.cleaned_texts_tokenized]

    def _train_tfidf(self):
        model, vectors = train_tfidf(
            self.cleaned_texts,
            self.tfidf_max_features,
            return_vectors=True
        )
        self.tfidf_model = model
        self.coprus_tfidf_vectors = vectors
        self.tfidf_features = model.get_feature_names()

    def _get_default_vector(self):
        return np.zeros(self.word_vector_size, dtype=np.float32)

    def _get_tfidf_vector_by_idx(self, doc_idx):
        return self.coprus_tfidf_vectors[doc_idx]

    def _get_doc_word_weights(self, doc, doc_tfidf_vector):
        word_weights = get_tfidf_word_weights(doc_tfidf_vector, self.tfidf_features)
        if word_weights == {}:
            return

        doc_word_weights = [word_weights.get(word, 0) for word in doc]
        doc_word_weights = np.array(list(doc_word_weights))
        return doc_word_weights

    def _get_doc_word_embed(self, doc):
        doc_word_vectors = [self.word_vector_dict.get(word, self._get_default_vector()) for word in doc]
        doc_word_vectors = list(doc_word_vectors)

        if len(doc_word_vectors) == 0:
            return np.zeros((10, self.word_vector_size), dtype=np.float32)

        doc_vector_arr = np.array(doc_word_vectors)
        return doc_vector_arr

    def _get_word_weight_matrix(self):
        word_weight_matrix = [self._get_doc_word_weights(word, self._get_tfidf_vector_by_idx(i))
                              for i, word in enumerate(self.cleaned_texts_tokenized)]
        self.word_weight_matrix = word_weight_matrix

    def _get_word_embed_matrix(self):
        word_embed_matrix = map(self._get_doc_word_embed, self.cleaned_texts_tokenized)
        self.word_embed_matrix = word_embed_matrix

    def _get_doc_embeds(self):
        doc_embeddings = [np.average(word_vector, axis=0, weights=weight)
                          for word_vector, weight in zip(self.word_embed_matrix, self.word_weight_matrix)]

        doc_embeddings = np.array(doc_embeddings)
        self.doc_embeddings = doc_embeddings.astype(np.float32)

    def _get_embed_single_doc(self, doc, doc_tfidf_vec):
        word_weights = self._get_doc_word_weights(doc, doc_tfidf_vec)
        word_embeds = self._get_doc_word_embed(doc)

        doc_embed = np.average(word_embeds, axis=0, weights=word_weights)
        doc_embed = doc_embed.astype(np.float32)
        return doc_embed

    def get_query_embedding(self, query_text):
        query_cleaned = self._preprocess_doc(query_text, self.use_stemming)

        query_tfidf_vec = self.tfidf_model.transform([' '.join(query_cleaned)])

        query_embed = self._get_embed_single_doc(query_cleaned, query_tfidf_vec)
        query_embed = query_embed

        return query_embed

    def fit(self):
        self._preprocess_texts()
        self._train_tfidf()
        self._get_word_weight_matrix()
        self._get_word_embed_matrix()

        self._get_doc_embeds()
