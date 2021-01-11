import re
import string
import time
from joblib import Parallel, delayed
from dataclasses import dataclass
from tqdm.auto import tqdm
import spacy
from spacy.util import minibatch
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer

tqdm.pandas()

RE_NUMERIC = re.compile(r"[0-9]+", re.UNICODE)
RE_PUNCT = re.compile(r'([%s])+' % re.escape(string.punctuation), re.UNICODE)
RE_WHITESPACE = re.compile(r"(\s)+", re.UNICODE)


def truncate_to_n_words(text, n_words):
    return ' '.join(text.split()[:n_words])


def _check_nltk():
    try:
        sent_tokenize('hello world')
    except LookupError:
        import nltk
        nltk.download('punkt')


@dataclass
class PreprocessedText:
    original_text: str
    preprocessed_text: str


class TextCleaner:
    @staticmethod
    def _clean_sentence(text):
        text = text.lower()
        text = RE_NUMERIC.sub('', text)
        text = RE_PUNCT.sub(' ', text)
        text = RE_WHITESPACE.sub(' ', text)
        if len(text) > 0 and text[0] == ' ':
            text = text[1:]
        if len(text) > 0 and text[-1] == ' ':
            text = text[:-1]
        return text

    def _clean_text(self, text):
        sentences = sent_tokenize(text)
        cleaned = [self._clean_sentence(sentence) for sentence in sentences]
        cleaned = '. '.join(cleaned)
        return cleaned

    def fit(self):
        return self

    def transform(self, texts, show_progress=True):
        if show_progress:
            text_iter = tqdm(texts)
        else:
            text_iter = texts

        return [self._clean_text(text) for text in text_iter]


class Lemmatizer:
    def __init__(self, model_name='en_core_web_sm', batch_size=1000, n_jobs=4, remove_pronouns=True):
        self.nlp = spacy.load(model_name, disable=['parser', 'ner'])
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.remove_pronouns = remove_pronouns

    @staticmethod
    def _check_pronouns(tokens, remove_pronouns):
        if remove_pronouns:
            tokens = [token for token in tokens if token != '-PRON-']
        return tokens

    def lemmatize_text(self, text):
        doc = self.nlp(text)
        lemmas = [token.lemma_ for token in doc]
        lemmas = self._check_pronouns(lemmas, self.remove_pronouns)
        lemmatized = ' '.join(lemmas)
        return lemmatized

    def _process_batch(self, batch):
        result = [self.lemmatize_text(text) for text in batch]
        return result

    def transform(self, texts):
        partitions = minibatch(texts, size=self.batch_size)
        executor = Parallel(n_jobs=self.n_jobs, backend="multiprocessing", prefer="processes")

        do = delayed(self._process_batch)
        tasks = (do(batch) for i, batch in enumerate(partitions))
        start_time = time.time()
        res = executor(tasks)
        res = [item for items in res for item in items]
        print(f'Elapsed: {time.time() - start_time}')
        return res


class Stemmer:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def fit(self):
        return self

    def _stem_sentence(self, sentence):
        words = sentence.split()
        stems = [self.stemmer.stem(word) for word in words]
        stemmed_sentence = ' '.join(stems)
        return stemmed_sentence

    def _stem_text(self, text):
        sentences = text.split('.')
        stemmed = [self._stem_sentence(sentence) for sentence in sentences]
        stemmed_text = '. '.join(stemmed)
        return stemmed_text

    def transform(self, texts, show_progress=True):
        if show_progress:
            text_iter = tqdm(texts)
        else:
            text_iter = texts

        return [self._stem_text(text) for text in text_iter]
