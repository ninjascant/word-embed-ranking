#!/bin/bash

CORPUS_URL=https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz

TRAIN_QREL_URL=https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz
TRAIN_QUERY_URL=https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz

DEV_QREL_URL=https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz
DEV_QUERY_URL=https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz

mkdir -p raw_data

help_str='Usage: cmd [-h] [-c]\n-c: download full corpus\n-q: download train and dev query and qrels datasets'

while getopts ":hcq" opt; do
  case ${opt} in
    h )
      echo -e $help_str
      ;;
    c ) # process option c
      echo -e "Downloading corpus\n"
      curl $CORPUS_URL -o raw_data/docs.tsv.gz
      ;;
    q)
      echo -e "Downloading queries and qrels\n"
      curl $TRAIN_QREL_URL -o raw_data/qrels_train.tsv.gz
      curl $TRAIN_QUERY_URL -o raw_data/queries_train.tsv.gz

      curl $DEV_QREL_URL -o raw_data/qrels_dev.tsv.gz
      curl $DEV_QUERY_URL -o raw_data/queries_dev.tsv.gz
      ;;
    \? ) echo -e $help_str
      ;;
  esac
done