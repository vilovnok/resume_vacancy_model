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
from .dataset import ListwiseContrastiveDataset
from .evaluator import Evaluator, run_evaluation
from .loss import ContrastiveLossWrapper
from .model import BiEncoderWrapper
from .trainer import ContrastiveTrainer
from .utils import load_beir_split


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"



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
            logger.info("Loading raw baseline model from HuggingFace")
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

            print()
            print(best_model_path)
            print()
            
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
        train_dataset = ListwiseContrastiveDataset(
            queries=queries,
            corpus=corpus,
            qrels=qrels,
            negative_sample_count=args.num_negatives,
            seed=args.seed
        )        

        # TODO: для тестирования работы кода
        # from torch.utils.data import Subset
        # max_samples = 100
        # train_dataset = Subset(train_dataset, range(max_samples))
        # TODO: конец тестирования
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        args.tokenizer = tokenizer

        teacher = BiEncoderWrapper.from_pretrained("./checkpoints/stage1/deepvk_RuModernBERT_base_v1/best_model/", args=args)                  
        
        args.model_name = args.student_model_name
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        args.tokenizer = tokenizer
        student = BiEncoderWrapper(args)

        loss_fn = ContrastiveLossWrapper(args)
        collate_fn_teacher = ListwiseCollator(
            tokenizer=teacher.tokenizer, 
            max_length_query=args.max_length_query, 
            max_length_passage=args.max_length_passage, 
        )
        collate_fn_student = ListwiseCollator(
            tokenizer=student.tokenizer, 
            max_length_query=args.max_length_query, 
            max_length_passage=args.max_length_passage, 
        )
        trainer = ContrastiveTrainer(
            teacher=teacher,
            student=student,
            args=args,
            train_dataset=train_dataset,
            collate_fn_student=collate_fn_student,
            collate_fn_teacher=collate_fn_teacher,
            loss_fn=loss_fn,
            allow_pre_train=False
        )
        trainer.train()
        logger.info("Bi-encoder Training completed.")

if __name__ == "__main__":
    main()
