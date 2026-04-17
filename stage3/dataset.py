from typing import List, Dict, Set, Tuple, Optional, Any, Union
import random
from dataclasses import dataclass
from torch.utils.data import Dataset
from tqdm import tqdm

import pandas as pd
import numpy as np



class BaseListwiseDataset(Dataset):
    def __init__(
        self,
        queries: Dict[str, Dict[str, str]],
        corpus: Dict[str, Dict[str, str]],
        qrels: Dict[str, Dict[str, int]],
        negative_sample_count: int = 9,
        seed: Optional[int] = None,
        device: str = 'cuda'
    ):
        self.queries = queries
        self.corpus = corpus
        self.qrels = qrels
        self.k = int(negative_sample_count)
        self.rng = np.random.default_rng(seed)

        self.all_ids = np.array(list(self.corpus.keys()), dtype=object)
        self.qrel_sets = {}
        self.neg_map = {}
        self.pos_pairs = []

        self._build_index_structures()

    def _build_index_structures(self):
        for qid, rels in self.qrels.items():
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

    def __len__(self):
        return len(self.pos_pairs)

    def _sample_negative_ids(self, qid: str, k: int) -> List[str]:
        pos_set = self.qrel_sets.get(qid, set())
        explicit_negs = self.neg_map.get(qid, [])

        neg_ids = set()

        if explicit_negs:
            if len(explicit_negs) >= k:
                return self.rng.choice(explicit_negs, size=k, replace=False).tolist()
            neg_ids.update(explicit_negs)

        while len(neg_ids) < k:
            pid = self.rng.choice(self.all_ids)
            if pid in pos_set or pid in neg_ids:
                continue
            neg_ids.add(pid)

        return list(neg_ids)

    def build_query_text(self, qid: str) -> str:
        raise NotImplementedError

    def build_passage_text(self, pid: str) -> str:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        qid, pos_pid = self.pos_pairs[idx]

        query_text = self.build_query_text(qid)
        pos_text = self.build_passage_text(pos_pid)

        neg_ids = self._sample_negative_ids(qid, self.k)
        neg_texts = [self.build_passage_text(pid) for pid in neg_ids]

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



class JFListwiseDataset(BaseListwiseDataset):
    def build_query_text(self, qid: str) -> str:
        x = self.queries[qid]
        return (
            f"Должность: {x.get('title', '')}\n"
            f"Описание: {x.get('description', '')}\n"
            f"Навыки: {x.get('skills', '')}"
        )

    def build_passage_text(self, pid: str) -> str:
        x = self.corpus[pid]
        return (
            f"Должность: {x.get('title', '')}\n"
            f"Описание: {x.get('description', '')}\n"
            f"Навыки: {x.get('skills', '')}"
        )

class JDListwiseDataset(BaseListwiseDataset):
    def build_query_text(self, qid: str) -> str:
        x = self.queries[qid]
        return f"Описание: {x.get('description', x.get('short_description'))}"

    def build_passage_text(self, pid: str) -> str:
        x = self.corpus[pid]
        return f"Описание: {x.get('description', x.get('short_description'))}"


class JSListwiseDataset(BaseListwiseDataset):
    def build_query_text(self, qid: str) -> str:
        x = self.queries[qid]
        return (
            f"Должность: {x.get('title', '')}\n"
            f"Навыки: {x.get('skills', '')}"
        )

    def build_passage_text(self, pid: str) -> str:
        x = self.corpus[pid]
        return (
            f"Должность: {x.get('title', '')}\n"
            f"Навыки: {x.get('skills', '')}"
        )
