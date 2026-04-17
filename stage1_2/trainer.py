import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluator import Evaluator, run_evaluation
from .utils import load_beir_split



class ContrastiveTrainer:
    """
    Trainer for our contrastive learning pipeline
    """
    
    def __init__(
        self,
        model: nn.Module,
        args,
        train_dataset,
        allow_pre_train,
        collate_fn=None,
        loss_fn=None
        ):
        
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.collate_fn = collate_fn
        self.loss_fn = loss_fn
        self.pre_train = allow_pre_train

        self.patience = getattr(args, 'early_stopping_patience', 1)
        self.best_score = 0.0  
        self.epochs_without_improvement = 0     
        
        self.val_data = None
        self.evaluator = None
        self.top_k = getattr(args, 'top_k', 5)
        
        if self.pre_train:
            self.learning_rate = args.learning_rate_pre_train
        else:
            self.learning_rate = args.learning_rate_main

        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision='fp16' if getattr(args, 'fp16', False) else 'no',
            log_with=getattr(args, 'log_with', None),
            project_dir=args.log_dir
        )

        self.setup_logging()

        self.lab_to_exclude = None
        
        if getattr(args, 'seed', None):
            set_seed(args.seed)

        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
            
        self.train_dataloader = self.create_dataloader(train_dataset, shuffle=True)
        
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_lr_scheduler()
            
        objs = [self.model,
                self.optimizer,
                self.train_dataloader,
                self.lr_scheduler]

        prepared = self.accelerator.prepare(*objs)

        (self.model,
         self.optimizer,
         self.train_dataloader,
         self.lr_scheduler
         ) = prepared
            
        # validation
        self.valid_data   = None
        self.evaluator  = None
        
        if getattr(args, "data_path", None):
            self.setup_validation(args.data_path)

        self.global_step = 0
        self.start_epoch = 0

    
    def setup_logging(self):
        """
        The function to track model training
        """          
            
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        safe_args = {
            k: v for k, v in vars(self.args).items()
            if isinstance(v, (int, float, str, bool))
        }
        
        if self.accelerator.is_main_process:
            if self.args.log_with == 'tensorboard':
                run_name = getattr(self.args, 'run_name', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}") 
                project_name = os.path.join('tensorboard', run_name)

            self.accelerator.init_trackers(
                project_name=project_name,
                config=safe_args
            )
            self.logger.info(f"TensorBoard logging initialized. Run 'tensorboard --logdir {self.args.log_dir}/tensorboard' to view")

    
    def create_dataloader(self, dataset, shuffle=True):
        if dataset is None:
            return None    

        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=getattr(self.args, 'num_workers', 4),
            pin_memory=True,
            drop_last=shuffle 
        )
        
    def create_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
    def create_lr_scheduler(self):
        """Create learning rate scheduler with optional warmup."""
        steps_per_epoch = len(self.train_dataloader)
        num_training_steps = steps_per_epoch * self.args.num_epochs
        warmup_pct = getattr(self.args, 'warmup_percent', 0.0)
        num_warmup_steps = int(num_training_steps * warmup_pct)
        
        print(f"Training setup:")
        print(f"  Total epochs: {self.args.num_epochs}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total steps: {num_training_steps}")
        print(f"  Warmup: {warmup_pct*100:.1f}% = {num_warmup_steps} steps")
        
        if num_warmup_steps == 0:
            if getattr(self.args, 'use_cosine_schedule', True):
                return CosineAnnealingLR(
                    self.optimizer,
                    T_max=num_training_steps,
                    eta_min=self.learning_rate * getattr(self.args, 'final_lr_percent', 0.1)
                )
            else:
                return ConstantLR(self.optimizer, factor=1.0, total_iters=num_training_steps)
            
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=getattr(self.args, 'warmup_start_percent', 0.1),  
            end_factor=1.0,
            total_iters=num_warmup_steps
        )  
        
        if getattr(self.args, 'use_cosine_schedule', True):
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - num_warmup_steps,
                eta_min=self.learning_rate * getattr(self.args, 'final_lr_percent', 0.1)
            )
        else:
            main_scheduler = ConstantLR(
            self.optimizer, 
            factor=1.0, 
            total_iters=num_training_steps - num_warmup_steps
        )
            
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[num_warmup_steps]
        )



    def _early_stopping(self, current_score: float) -> bool:
        """
        Return True  → halt training
               False → continue
        """
        if current_score > self.best_score:
            self.best_score = current_score
            self.epochs_without_improvement = 0

            if self.accelerator.is_main_process:
                best_dir = os.path.join(self.args.checkpoint_dir, "best_model")
                
                if os.path.exists(best_dir):
                    shutil.rmtree(best_dir)
                    
                self.save_checkpoint("best_model")
                self.logger.info("New best model saved.")

            return False

        self.epochs_without_improvement += 1
        self.logger.info(
            f"No improvement for {self.epochs_without_improvement} epoch(s). "
            f"Best so far: {self.best_score:.4f}"
        )
        return self.epochs_without_improvement >= self.patience

    
    def setup_validation(self, data_path: str):
        corpus, queries, qrels = load_beir_split(data_path, 'test')
        self.valid_data = (corpus, queries, qrels)
        
        self.evaluator = Evaluator(
            model=self.accelerator.unwrap_model(self.model), 
            device=self.accelerator.device,
            args=self.args
        )
            
    def train(self):
        """Main training loop."""
        
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataset)}")
        self.logger.info(f"  Num Epochs = {self.args.num_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {len(self.train_dataloader) * self.args.num_epochs}")
        self.logger.info(f"  Current Seed = {self.args.seed}")
        
        
        if self.valid_data:
            self.logger.info(f"  Validation samples = {len(self.valid_data[-1])}")
            self.logger.info(f"  Early stopping patience = {self.patience}")
        
        
        if self.args.save_baseline and self.accelerator.is_main_process:
            self.save_checkpoint("baseline_initial")
        
        for epoch in range(self.start_epoch, self.args.num_epochs):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            self.logger.info(f"{'='*50}")

            train_metrics = self.train_epoch(epoch)

            if self.accelerator.is_main_process:
                self.accelerator.log(train_metrics, step=self.global_step)
                
            if self.valid_data:                
                self.evaluator.refresh_model(
                    self.accelerator.unwrap_model(self.model)
                )
                self.logger.info(f"[EVAL] Running Evaluation...")

                r5, mrr = run_evaluation(
                    evaluator=self.evaluator, 
                    data=self.valid_data, 
                    top_k=self.top_k
                )
                
                if self.accelerator.is_main_process:
                    self.logger.info(
                        f"EVAL  R-Precision@{self.top_k} = {r5:.4f}"
                    )
                    self.logger.info(
                        f"EVAL MRR = {mrr:.4f}"
                    )
                    self.accelerator.log(
                        {f"eval/rprecision@{self.top_k}": r5}, 
                        step=self.global_step
                    )

                if self._early_stopping(r5):
                    self.logger.info("Early stopping triggered.")
                    break
    
            if (epoch + 1) % self.args.save_freq == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}")
                
        self.logger.info("Training completed!")
        if getattr(self.args, "data_path", None):                
            self.logger.info("[TEST] Running final test...")

            data_path = self.args.data_path
            if not data_path:  
                return

            test_data = load_beir_split(data_path, 'test')


            if self.args.fp16: 
                self.model.half()
            
            self.evaluator.refresh_model(
                self.accelerator.unwrap_model(self.model)
            )        

            r5, mrr = run_evaluation(
                evaluator=self.evaluator, 
                data=test_data, 
                top_k=self.top_k
            )
            
            if self.accelerator.is_main_process:
                self.accelerator.log(
                    {f"test/rprecision@{self.top_k}": r5},
                    step=self.global_step,
                )
                self.logger.info(
                    f"TEST R-Precision@{self.top_k} = {r5:.4f}"
                )
                self.logger.info(
                    f"TEST MRR = {mrr:.4f}"
                )

                self.evaluator.compute_final_metrics()

    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Training for a single epoch, returns metrics"""
        
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
 
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training epoch {epoch + 1}",
            disable=not self.accelerator.is_local_main_process
        )
        for batch in progress_bar:
            with self.accelerator.accumulate(self.model):
                loss_dict = self.training_step(batch)
                loss = loss_dict['loss']
                
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients and getattr(self.args, 'max_grad_norm', None):
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )
               
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                gathered_loss = self.accelerator.gather(loss).mean().item()
                epoch_loss += gathered_loss
                num_batches += 1
               
                progress_bar.set_postfix({
                    'loss': gathered_loss,
                    'lr': self.optimizer.param_groups[0]['lr']
                })

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    if self.global_step % self.args.log_freq == 0:
                        self.accelerator.log({
                            'train/loss_step': gathered_loss,
                            'train/lr': self.optimizer.param_groups[0]['lr'],
                        }, step=self.global_step)

        return {
            'train/loss_epoch': epoch_loss / num_batches,
            'train/epoch': epoch + 1
        }

        
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Single training step."""

        anchor_embeddings = self.model(
            batch['anchor_input_ids'],
            batch['anchor_attention_mask']
        )
        positive_embeddings = self.model(
            batch['positive_input_ids'],
            batch['positive_attention_mask']
        )
        
        negative_embeddings = None
        if batch['negative_input_ids'] is not None:
            neg_ids = batch['negative_input_ids']
            neg_mask = batch['negative_attention_mask']
            batch_size, num_negs, seq_len = neg_ids.shape
            
            neg_ids = neg_ids.view(-1, seq_len)
            neg_mask = neg_mask.view(-1, seq_len)
            neg_embeddings_flat = self.model(neg_ids, neg_mask)
            negative_embeddings = neg_embeddings_flat.view(batch_size, num_negs, -1)
        
        loss_output = self.loss_fn(
            anchor_embeddings=anchor_embeddings,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings,
            in_batch_negative_masks=batch.get('in_batch_negative_masks')
        )
        return {"loss": loss_output}

        
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        if not self.accelerator.is_main_process:
            return
        
        save_path = os.path.join(self.args.checkpoint_dir, self.args.run_name, name)
            
        os.makedirs(save_path, exist_ok=True)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(save_path)
        unwrapped_model.tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
        
        trainer_state = {
            'epoch': self.global_step // len(self.train_dataloader),
            'global_step': self.global_step,
            'best_metric': self.best_score,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'args': vars(self.args),
        }
        
        torch.save(trainer_state, os.path.join(save_path, 'trainer_state.pt'))
        self.logger.info(f"Checkpoint saved to {save_path}")