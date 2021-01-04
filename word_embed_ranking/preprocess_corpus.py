import re
import string
import argparse
from tqdm.auto import tqdm
import pandas as pd
import spacy
from nltk.tokenize import sent_tokenize

tqdm.pandas()

# nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

RE_NUMERIC = re.compile(r"[0-9]+", re.UNICODE)
RE_PUNCT = re.compile(r'([%s])+' % re.escape(string.punctuation), re.UNICODE)


def clean_text(text):
    text = text.lower()
    text = RE_NUMERIC.sub('', text)
    text = RE_PUNCT.sub('', text)
    # doc = nlp(text)
    # tokens = [token for token in doc if not token.is_punct]# [:1_000]
    return text


def lemmatize_text(tokens):
    return " ".join([token.lemma_ for token in tokens if token.lemma_ != '-PRON-'])


def postprocess_text(lemmatized_text):
    spaces_cleaned = filter(lambda x: x != ' ', lemmatized_text.split())
    return ' '.join(spaces_cleaned)


def process_text(text):
    sentences = sent_tokenize(text)
    cleaned = [clean_text(sentence) for sentence in sentences]
    cleaned = '. '.join(cleaned)
    # cleaned = clean_text(text)
    # lemmatized = lemmatize_text(cleaned)
    # postprocessed = postprocess_text(lemmatized)
    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus-file', type=str)
    parser.add_argument('--outfile', type=str, default='./corpus_cleaned_train.csv')
    parser.add_argument('--sample-size', type=int, default=None)
    args = parser.parse_args()

    if not args.sample_size:
        texts = pd.read_csv(args.corpus_file)
    else:
        texts = pd.read_csv(args.corpus_file, nrows=args.sample_size)

    texts.loc[texts['text'].isna(), 'text'] = ''

    texts['text'] = texts['text'].progress_apply(process_text)
    texts['query'] = texts['query'].progress_apply(process_text)
    texts.loc[texts['text'].isna(), 'text'] = ''
    texts.loc[texts['query'].isna(), 'text'] = ''

    texts.to_csv(args.outfile, index=False)


if __name__ == '__main__':
    main()
