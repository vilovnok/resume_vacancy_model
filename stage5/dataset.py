from typing import List, Dict, Set, Tuple, Optional, Any, Union
import random
from dataclasses import dataclass
from torch.utils.data import Dataset
from tqdm import tqdm

import pandas as pd
import numpy as np




class ListwiseContrastiveDataset(Dataset):

    def __init__(
        self,
        df: Optional["pandas.DataFrame"] = None,
        queries: Optional[Dict[str, str]] = None,
        corpus: Optional[Dict[str, Dict[str, str]]] = None,
        qrels: Optional[Dict[str, Dict[str, int]]] = None,
        negative_sample_count: int = 9,
        seed: Optional[int] = None,
        desc: str = "Building listwise dataset",
    ):

        self.k = int(negative_sample_count)
        self.rng = np.random.default_rng(seed)

        if df is not None:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("df must be a pandas.DataFrame")
            self._from_dataframe(df, desc=desc)
        else:
            if queries is None or corpus is None or qrels is None:
                raise ValueError(
                    "If df is not provided you must pass queries, corpus and qrels"
                )
            self.queries = queries
            self.corpus = corpus
            self.qrels = qrels
            self._build_index_structures(desc=desc)

    def _from_dataframe(self, df, desc: str = "Building from df"):

        required = {"query_id", "passage_id", "query", "passage", "label"}
        missing = required - set(df.columns)
        
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        self.queries = {
            str(pid): {"text": str(text) if pd.notna(text) else ""}
            for pid, text in df.groupby("query_id", sort=False)["query"].first().items()
        }
        self.corpus = {
            str(pid): {"text": str(text) if pd.notna(text) else ""}
            for pid, text in df.groupby("passage_id", sort=False)["passage"].first().items()
        }

        qrels = {}
        for row in df.itertuples(index=False):
            qid = str(row.query_id)
            pid = str(row.passage_id)
            label = int(row.label) if row.label is not None else 0
            qrels.setdefault(qid, {})[pid] = label

        self.qrels = qrels
        self._build_index_structures(desc=desc)

    def _build_index_structures(self, desc: str = "Building structures"):

        self.all_ids = np.array(list(self.corpus.keys()), dtype=object)
        self.id_to_idx = {pid: i for i, pid in enumerate(self.all_ids)}

        self.qrel_sets = {}     
        self.neg_map = {}        
        self.pos_pairs = []

        for qid, rels in tqdm(self.qrels.items(), desc=desc):

            pos_ids = []
            neg_ids = []

            for pid, rel in rels.items():
                if int(rel) > 0:
                    pos_ids.append(pid)
                else:
                    neg_ids.append(pid)

            if not pos_ids:
                continue

            self.qrel_sets[qid] = set(pos_ids)
            self.neg_map[qid] = neg_ids

            for pid in pos_ids:
                self.pos_pairs.append((qid, pid))

        self.num_pairs = len(self.pos_pairs)

    def __len__(self) -> int:
        return self.num_pairs

    def _sample_negative_ids(self, qid: str, k: int) -> List[str]:

        pos_set = self.qrel_sets.get(qid, set())
        explicit_negs = self.neg_map.get(qid, [])

        neg_ids = set()

        if explicit_negs:
            if len(explicit_negs) >= k:
                return self.rng.choice(explicit_negs, size=k, replace=False).tolist()
            else:
                neg_ids.update(explicit_negs)

        remaining = k - len(neg_ids)
        if remaining <= 0:
            return list(neg_ids)

        available = len(self.all_ids) - len(pos_set) - len(neg_ids)
        if available <= 0:
            return list(neg_ids)

        attempt_batch = max(4 * remaining, 128)

        while len(neg_ids) < k:
            idxs = self.rng.integers(0, len(self.all_ids), size=attempt_batch)
            candidates = self.all_ids[idxs]

            for pid in candidates:
                if pid in pos_set:
                    continue
                if pid in neg_ids:
                    continue
                neg_ids.add(pid)
                if len(neg_ids) == k:
                    break

            if len(neg_ids) < k:
                attempt_batch = min(attempt_batch * 2, len(self.all_ids))

        return list(neg_ids)

    def __getitem__(self, idx: int):

        qid, pos_pid = self.pos_pairs[idx]

        query_text = self.queries[qid]["text"]
        pos_text = self.corpus[pos_pid]["text"]

        neg_ids = self._sample_negative_ids(qid, self.k)
        neg_texts = [self.corpus[nid]["text"] for nid in neg_ids]

        labels = [1] + [0] * len(neg_ids)

        return {
            "query_id": qid,
            "query": query_text,
            "positive_id": pos_pid,
            "positive": pos_text,
            "negative_ids": neg_ids,
            "negatives": neg_texts,
            "labels": labels,
        }
