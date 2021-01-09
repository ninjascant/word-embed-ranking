import re
import string
from dataclasses import dataclass
from tqdm.auto import tqdm
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
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


class SpacyTokenizer:
    def __init__(self):
        nlp = English()
        self.tokenizer = Tokenizer(nlp.vocab)

    def fit(self):
        return self

    def transform(self, text):
        tokens = self.tokenizer(text)
        return [token.text for token in tokens]


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


@dataclass
class LemmatizedText:
    original_text: str
    lemmatized_text: str


class Lemmatizer:
    def __init__(self, max_len=None, remove_pronouns=False):
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        self.max_len = max_len
        self.remove_pronouns = remove_pronouns

    def _truncate_texts(self, texts):
        return [truncate_to_n_words(text, self.max_len) for text in texts]

    def _check_pronouns(self, tokens):
        if self.remove_pronouns:
            tokens = [token for token in tokens if token[1] != '-PRON-']
        return tokens

    def _lemmatize_sentence(self, text):
        doc = self.nlp(text)
        tokens = [token for token in doc]
        tokens = [(token.text, token.lemma_) for token in tokens]
        tokens = self._check_pronouns(tokens)
        lemmas = ' '.join([token[1] for token in tokens])
        raw_text = ' '.join([token[0] for token in tokens])
        return raw_text, lemmas

    def _lemmatize_text(self, text):
        sentences = text.split('.')
        lemmatized_sentences = [self._lemmatize_sentence(sentence) for sentence in sentences]
        raw_text = '. '.join([item[0] for item in lemmatized_sentences])
        lemmatized_text = '. '.join([item[1] for item in lemmatized_sentences])
        return LemmatizedText(raw_text, lemmatized_text)

    def fit(self):
        return self

    def transform(self, texts, show_progress=True):
        if self.max_len:
            texts = self._truncate_texts(texts)
        if show_progress:
            text_iter = tqdm(texts)
        else:
            text_iter = texts

        texts = [self._lemmatize_text(text) for text in text_iter]
        return texts


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

    def transform(self, texts, show_progress=False):
        if show_progress:
            text_iter = tqdm(texts)
        else:
            text_iter = texts

        return [self._stem_text(text) for text in text_iter]
