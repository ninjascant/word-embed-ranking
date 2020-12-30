import re
from tqdm.auto import tqdm
import pandas as pd
import spacy

tqdm.pandas()

nlp = spacy.load('en', disable=['parser', 'ner'])

RE_NUMERIC = re.compile(r"[0-9]+", re.UNICODE)


def clean_text(text):
    text = text.lower()
    text = RE_NUMERIC.sub('', text)
    doc = nlp(text)
    tokens = [token for token in doc if not token.is_punct][:1_000]
    return tokens


def lemmatize_text(tokens):
    return " ".join([token.lemma_ for token in tokens if token.lemma_ != '-PRON-'])


def postprocess_text(lemmatized_text):
    spaces_cleaned = filter(lambda x: x != ' ', lemmatized_text.split())
    return ' '.join(spaces_cleaned)


def process_text(text):
    cleaned = clean_text(text)
    lemmatized = lemmatize_text(cleaned)
    return postprocess_text(lemmatized)


def main():
    texts = pd.read_csv('./corpus_with_queries_train_sample1.csv')
    texts = texts.loc[~texts['text'].isna()]
    # texts['text'] = texts['text'].progress_apply(process_text)
    # texts.to_csv('./corpus_cleaned_sample.csv', index=False)
    texts['query'] = texts['query'].progress_apply(process_text)
    texts[['query', 'qid', 'doc_id']].to_csv('./queries_cleaned_sample.csv', index=False)


if __name__ == '__main__':
    main()
