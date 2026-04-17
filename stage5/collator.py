from typing import List, Dict, Set, Tuple, Optional, Any
import torch



class ListwiseCollator:
    def __init__(
        self,
        tokenizer,
        max_length_query: int = 64,
        max_length_passage: int = 128,
        padding: str = "longest", 
        truncation: bool = True,
        pad_to_max_length: bool = False, 
        return_tensors: str = "pt",
        enable_monitoring: bool = False,
        in_batch_strategy: str = "in_batch",
    ):
        self.tokenizer = tokenizer
        self.max_length_query = max_length_query
        self.max_length_passage = max_length_passage
        self.padding = padding
        self.truncation = truncation
        self.pad_to_max_length = pad_to_max_length
        self.return_tensors = return_tensors
        self.enable_monitoring = enable_monitoring
        self.in_batch_strategy = in_batch_strategy

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)

        anchors = []
        positives = []
        negatives_lists = []
        query_ids = []
        pos_ids = []
        neg_ids = []

        for ex in batch:

            if "anchor" in ex:
                anchors.append(ex["anchor"])
            elif "query" in ex:
                anchors.append(ex["query"])
            else:
                raise KeyError("Each example must contain 'anchor' or 'query' key")

            if "positive" in ex:
                positives.append(ex["positive"])
            else:
                raise KeyError("Each example must contain 'positive' key")

            negs = ex.get("negatives") or ex.get("negative_texts") or ex.get("negative_ids") or []

            if negs is None:
                negs = []
            negatives_lists.append(list(negs))

            # ids (optional)
            query_ids.append(ex.get("query_id") or ex.get("anchor_id") or None)
            pos_ids.append(ex.get("positive_id") or None)
            neg_ids.append(ex.get("negative_ids") or None)

        anchor_enc = self.tokenizer(
            anchors,
            max_length=self.max_length_query,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors
        )
        positive_enc = self.tokenizer(
            positives,
            max_length=self.max_length_passage,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors
        )

        neg_counts = [len(n) for n in negatives_lists]
        max_neg = max(neg_counts) if neg_counts else 0

        neg_input_ids = None
        neg_attention_mask = None
        neg_input_ids_flat = None

        monitoring_info = None

        if max_neg == 0:
            neg_counts_tensor = torch.zeros(B, dtype=torch.long)
        
        else:
            padded_neg_texts = []
            for negs in negatives_lists:
                row = list(negs)
                if len(row) < max_neg:
                    row.extend([""] * (max_neg - len(row)))
                padded_neg_texts.extend(row)

            neg_enc = self.tokenizer(
                padded_neg_texts,
                max_length=self.max_length_passage,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors=self.return_tensors
            )

            seq_len = neg_enc["input_ids"].size(1)
            neg_input_ids = neg_enc["input_ids"].view(B, max_neg, seq_len)
            neg_attention_mask = neg_enc["attention_mask"].view(B, max_neg, seq_len)
            neg_input_ids_flat = neg_enc["input_ids"]
            neg_counts_tensor = torch.tensor(neg_counts, dtype=torch.long)

            if self.enable_monitoring:
                monitoring_info = {
                    "min_negatives": int(neg_counts_tensor.min().item()),
                    "max_negatives": int(neg_counts_tensor.max().item()),
                    "avg_negatives": float(neg_counts_tensor.float().mean().item())
                }

        if self.in_batch_strategy == "none":
            in_batch_mask = torch.zeros(B, B, dtype=torch.bool)
        else:
            in_batch_mask = torch.ones(B, B, dtype=torch.bool)
            in_batch_mask.fill_diagonal_(False)

        out: Dict[str, Any] = {
            "anchor_input_ids": anchor_enc["input_ids"],
            "anchor_attention_mask": anchor_enc["attention_mask"],
            "positive_input_ids": positive_enc["input_ids"],
            "positive_attention_mask": positive_enc["attention_mask"],
            "negative_input_ids": neg_input_ids,                 
            "negative_attention_mask": neg_attention_mask,       
            "negative_input_ids_flat": neg_input_ids_flat,       
            "negative_counts": neg_counts_tensor if max_neg > 0 else torch.zeros(B, dtype=torch.long),
            "in_batch_negative_masks": in_batch_mask,            
            "query_ids": query_ids,
            "pos_ids": pos_ids,
            "neg_ids": neg_ids,
            "monitoring_info": monitoring_info
        }

        return out