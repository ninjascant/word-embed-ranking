import re
import argparse
from tqdm.auto import tqdm
import pandas as pd
import spacy

tqdm.pandas()

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus-file', type=str)
    parser.add_argument('--outfile', type=str, default='./corpus_cleaned_train.csv')
    args = parser.parse_args()
    print(args)

    texts = pd.read_csv(args.corpus_file)
    texts = texts.loc[~texts['text_truncated'].isna()]
    print(texts.shape[0])

    texts['text_truncated'] = texts['text_truncated'].progress_apply(process_text)
    texts['query'] = texts['query'].progress_apply(process_text)

    texts.to_csv(args.outfile, index=False)


if __name__ == '__main__':
    main()
