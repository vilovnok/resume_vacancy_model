import os
import json



def load_beir_split(data_dir: str, split: str):
    allowed = {"train", "valid", "test"}
    if split not in allowed:
        raise ValueError(f"split must be one of {allowed}, got {split}")

    queries_path = os.path.join(data_dir, split, "queries.json")
    corpus_path = os.path.join(data_dir, split, "corpus.json")
    qrels_path = os.path.join(data_dir, split, "qrels.json")

    for path in [queries_path, corpus_path, qrels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels = json.load(f)

    return corpus, queries, qrels