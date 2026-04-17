import argparse
import json
import logging
import os
import pathlib
from pathlib import Path
import random
import warnings
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.models import SentenceBERT
from beir.retrieval.search.dense import (
    DenseRetrievalExactSearch,
    FlatIPFaissSearch,
)

from .model import BiEncoderWrapper
from .utils import load_beir_split




class Evaluator:
    """
    Evaluator для задачи matching резюме и вакансий.
    Принимает на вход queries (резюме), corpus (вакансии) и qrels (позитивные вакансии на query).
    """

    def __init__(self, model, args, device="cuda", batch_size=256):
        self.model = model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.corpus_embeddings: Optional[torch.Tensor] = None
        self.corpus_ids: Optional[List[str]] = None
        self.args = args


    def refresh_model(self, model) -> None:
        self.model = model

    
    def encode_corpus(self, corpus: Dict[str, Dict[str, str]]):
        self.corpus_ids = list(corpus.keys())
        texts = [corpus[doc_id]["text"] for doc_id in self.corpus_ids]
        self.corpus_embeddings = self._encode_texts(texts, "corpus")
        self.corpus_embeddings = F.normalize(self.corpus_embeddings, dim=-1)

    
    def evaluate(self, queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], k: int = 5, return_preds: bool = False, output_path: Optional[str] = None):
        assert self.corpus_embeddings is not None, "Corpus must be encoded first!"

        query_ids = list(queries.keys())
        query_texts = [queries[qid]['text'] for qid in query_ids]
        query_embeddings = self._encode_texts(query_texts, "queries")
        query_embeddings = F.normalize(query_embeddings, dim=-1)

        sim_matrix = torch.matmul(query_embeddings, self.corpus_embeddings.T)

        rprecisions = []
        mrr_scores = []
        preds_all = [] if return_preds else None

        for i, qid in enumerate(query_ids):
            sims = sim_matrix[i]
            top_k = min(k, len(self.corpus_ids))
            top_idx = torch.topk(sims, top_k).indices
            top_idx_all = torch.topk(sims, min(100, len(self.corpus_ids))).indices

            predicted_docs = [self.corpus_ids[j.item()] for j in top_idx]
            predicted_docs_all = [self.corpus_ids[j.item()] for j in top_idx_all]

            positives = set(qrels.get(qid, {}).keys())
            if not positives:
                continue

            relevant_in_topk = sum(1 for doc in predicted_docs if doc in positives)
            rprecision = relevant_in_topk / min(len(positives), k)
            rprecisions.append(rprecision)

            mrr = 0.0
            for rank, doc in enumerate(predicted_docs_all, 1):
                if doc in positives:
                    mrr = 1.0 / rank
                    break
            mrr_scores.append(mrr)

            if return_preds:
                pred_with_scores = [(self.corpus_ids[j.item()], sims[j].item()) for j in top_idx]
                preds_all.append({
                    "query_id": qid,
                    "predicted_docs": pred_with_scores,
                    "true_docs": list(positives),
                    "rprecision": rprecision,
                    "mrr": mrr
                })

        out = {
            f"rprecision@{k}": float(np.mean(rprecisions)),
            "mrr": float(np.mean(mrr_scores))
        }

        if return_preds:
            out["predictions"] = preds_all
            if output_path:
                with open(output_path, "w") as f:
                    json.dump(preds_all, f, indent=2)
        return out

    
    def _encode_texts(self, texts: List[str], desc: str)-> torch.Tensor:
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc=f"Encoding {desc}"):
                batch = texts[i:i+self.batch_size]
                emb = self.model.encode(batch, convert_to_numpy=True, device=self.device, show_progress_bar=False, normalize_embeddings=False)
                emb = torch.tensor(emb, device=self.device)
                emb = F.normalize(emb, p=2, dim=-1)
                all_embeddings.append(emb)
        return torch.cat(all_embeddings, dim=0).to(self.device)



    def compute_final_metrics(self, prefix: str="ft", device: str="cuda") -> None:
        """
        Используем одну модель для кодирования документов и запросов.
        Строим FAISS индекс и оцениваем retrieval на BEIR.
        """
        
        self.model.eval()
        os.makedirs(self.args.result_path, exist_ok=True)
        filename = os.path.join(self.args.result_path, f"{prefix}_{self.args.run_name}.csv")
        
        records = []
        ds_name = self.args.datasets[0]
        
        dir_raw = Path(self.args.data_path)
        dir_idx = Path(self.args.index_path)
        dir_idx.mkdir(parents=True, exist_ok=True)

        logging.info(f"Processing dataset: {ds_name}")

        corpus, queries, qrels = load_beir_split(
            data_dir=dir_raw, 
            split='test'
        )

        index_file_name = f"{prefix}_{self.args.model_name}.{self.args.extension}.faiss"
        index_path = dir_idx / index_file_name.replace('/','_')
        
        if index_path.exists():
            logging.info("Loading existing FAISS index...")
            index = faiss.read_index(str(index_path))
            doc_ids = np.load(dir_idx / f"{prefix}_doc_ids.npy", allow_pickle=True)
        else:
            logging.info("Creating new FAISS index...")

            doc_ids = list(corpus.keys())
            docs = [corpus[doc_id]['text'] for doc_id in doc_ids]
    
            doc_embeddings = self.model.encode(
                docs,
                batch_size=self.args.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
                device=device
            )

            dimension = doc_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(doc_embeddings)

            faiss.write_index(index, str(index_path))
            np.save(dir_idx / f"{prefix}_doc_ids.npy", doc_ids)
            logging.info(f"FAISS index created at {index_path}")

        query_ids = list(queries.keys())
        query_texts = [queries[qid]['text'] for qid in query_ids]

        query_embeddings = self.model.encode(
            query_texts,
            batch_size=self.args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            device=device
        )
        
        k_max = max(self.args.k_values)
        D, I = index.search(query_embeddings, k_max)

        results, record = {}, {}
        for q_idx, qid in enumerate(query_ids):
            doc_scores = {}
            for rank, doc_index in enumerate(I[q_idx]):
                doc_id = doc_ids[doc_index]
                score = float(D[q_idx][rank])
                doc_scores[doc_id] = score
            results[qid] = doc_scores

        retriever = EvaluateRetrieval(score_function="cos_sim")        
        metrics = retriever.evaluate(qrels, results, k_values=self.args.k_values)

        for metric in metrics:
            record.update(metric)

        record.update({
                "datetime": pd.Timestamp.now(),
                "dataset": ds_name.replace("/", "-"),
                "model": self.args.model_name
            })
        records.append(record)
        
        df = pd.DataFrame(records)
        df.to_csv(filename, index=False)
        logging.info(f"Saved evaluation results to {filename}")



def run_evaluation(evaluator: Evaluator, data: tuple, top_k: int):
    corpus, queries, qrels = data
    evaluator.encode_corpus(corpus)
    test_stats = evaluator.evaluate(
        queries, 
        qrels, 
        k=top_k, 
        return_preds=False 
    )
    r5   = test_stats[f"rprecision@{top_k}"]
    mrr  = test_stats[f"mrr"]
                    
    return r5, mrr
