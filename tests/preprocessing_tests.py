import unittest
from itertools import zip_longest
from word_embed_ranking.preprocess_corpus import TextCleaner, Lemmatizer, Stemmer


class TextCleanerTest(unittest.TestCase):
    @staticmethod
    def test_clean_sentences():
        with open('tests/raw_texts.txt') as file:
            sentences = file.readlines()[:2]
        with open('tests/cleaned_texts.txt') as file:
            correct_results = file.readlines()

        text_cleaner = TextCleaner()
        cleaned = text_cleaner.transform(sentences, show_progress=False)

        is_correct_res = [res[0].split() == res[1].split()
                          for res in zip_longest(cleaned, correct_results)]
        assert all(is_correct_res)

    @staticmethod
    def test_lemmatize_sentences():
        with open('tests/raw_texts.txt') as file:
            sentences = file.readlines()

        text_cleaner = TextCleaner()
        lemmatizer = Lemmatizer()

        cleaned = text_cleaner.transform(sentences, show_progress=False)
        preprocessed_texts = lemmatizer.transform(cleaned, show_progress=False)

        is_correct_len = [len(text.original_text.split()) == len(text.lemmatized_text.split())
                          for text in preprocessed_texts]
        assert all(is_correct_len)

    @staticmethod
    def test_remove_pronouns():
        with open('tests/raw_texts.txt') as file:
            sentences = file.readlines()

        text_cleaner = TextCleaner()
        cleaned = text_cleaner.transform(sentences, show_progress=False)

        lemmatizer_with_clean = Lemmatizer(remove_pronouns=True)
        cleaned_preprocessed_texts = lemmatizer_with_clean.transform(cleaned, show_progress=False)

        lemmatizer_without_clean = Lemmatizer(remove_pronouns=False)
        preprocessed_texts = lemmatizer_without_clean.transform(cleaned, show_progress=False)

        cleaned_has_not_pron = ['-PRON-' not in text.lemmatized_text for text in cleaned_preprocessed_texts]
        preprocessed_has_pron = ['-PRON-' in text.lemmatized_text for text in preprocessed_texts]
        assert all(cleaned_has_not_pron)
        assert any(preprocessed_has_pron)

    @staticmethod
    def test_stem_sentences():
        with open('tests/cleaned_texts.txt') as file:
            sentences = file.readlines()

        stemmer = Stemmer()
        stemmed = stemmer.transform(sentences, show_progress=False)

        is_correct_len = [len(item[0].split()) == len(item[1].split()) for item in zip_longest(sentences, stemmed)]
        assert all(is_correct_len)