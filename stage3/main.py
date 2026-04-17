import hashlib
import itertools
import json
import logging
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .arguments import create_parser
from .collator import ListwiseCollator
from .dataset import JFListwiseDataset, JDListwiseDataset, JSListwiseDataset
from .evaluator import Evaluator, run_evaluation
from .loss import ContrastiveLossWrapper
from .model import BiEncoderWrapper
from .trainer import Trainer
from .utils import load_beir_split


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"



def main():
    parser = create_parser()
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if args.test_only:
        assert args.data_path, "You must provide --data_path"
        assert args.model_name, "You must provide --model_name"
        
        logger.info("Running in test-only mode")

        prefix = None
        if args.pure_baseline:
            logger.info("Loading raw baseline model")
            prefix = "base"
            
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            args.tokenizer = tokenizer
            model = BiEncoderWrapper(args)
            
            model.encoder = AutoModel.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16 if getattr(args, "fp16", False) else torch.float32
            )
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        else:                
            logger.info("Loading fine-tuned model from checkpoint")
            prefix = "ft"
            best_model_path = os.path.join(args.checkpoint_dir, args.run_name, "best_model")

            tokenizer = AutoTokenizer.from_pretrained(os.path.join(best_model_path, "tokenizer"))
            args.tokenizer = tokenizer
            
            model = BiEncoderWrapper.from_pretrained(best_model_path, args=args)
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")

        test_data = load_beir_split(
            data_dir=args.data_path, 
            split='test'
        )
        
        logger.info(f"[TEST] Running test...")
        evaluator = Evaluator(model=model,  args=args)
        r5, mrr = run_evaluation(
            evaluator=evaluator, 
            data=test_data, 
            top_k=args.top_k, 
        )

        logger.info(
            f"TEST R-Precision@{args.top_k} = {r5:.4f}"
        )
        logger.info(
            f"TEST MRR = {mrr:.4f}"
        )
        evaluator.compute_final_metrics(prefix=prefix)
        logger.info("Testing completed.")
        
    
    if args.pre_train:
        logger.info(f"Initiating the Pre Training")
        logger.info(f"Loading data from {args.data_path}")
        
        corpus, queries, qrels = load_beir_split(
            data_dir=args.data_path,
            split="train"
        )
    
        train_dataset = {
            "jf": JFListwiseDataset(queries=queries, corpus=corpus, qrels=qrels, negative_sample_count=3, seed=22),
            "jd": JDListwiseDataset(queries=queries, corpus=corpus, qrels=qrels, negative_sample_count=3, seed=22),
            "js": JSListwiseDataset(queries=queries, corpus=corpus, qrels=qrels, negative_sample_count=3, seed=22),
        }

        # TODO: для тестирования работы кода
        # from torch.utils.data import Subset
        # max_samples = 1000
        # train_dataset['jf'] = Subset(train_dataset['jf'], range(max_samples))
        # train_dataset['jd'] = Subset(train_dataset['jd'], range(max_samples))
        # train_dataset['js'] = Subset(train_dataset['js'], range(max_samples))
        # TODO: конец тестирования
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        args.tokenizer = tokenizer

        model = BiEncoderWrapper(args)

        loss_fn = ContrastiveLossWrapper(args)
        collate_fn = ListwiseCollator(
            tokenizer=model.tokenizer, 
            max_length_query=args.max_length_query, 
            max_length_passage=args.max_length_passage, 
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            collate_fn=collate_fn,
            loss_fn=loss_fn,
            allow_pre_train=False
        )
        trainer.train()
        logger.info("Bi-encoder Training completed.")

if __name__ == "__main__":
    main()
