# word-embed-ranking

## Scripts

1. `./scripts/download.sh`
2. `./scripts/unzip_data.sh`
3. `python3 ./scripts/get_data_subset.py --type=train`
4. `python3 ./scripts/get_data_subset.py --type=dev`
5. `python3 ./scripts/join_docs_with_queries.py`
6. `python3 truncate_texts.py`