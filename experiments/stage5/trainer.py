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


from .model import HiddenProjection
import torch.nn.functional as F


class ContrastiveTrainer:
    """
    Trainer for our contrastive learning pipeline
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,        
        args,
        train_dataset,
        allow_pre_train,
        collate_fn_teacher=None,
        collate_fn_student=None,
        loss_fn=None
        ):
        
        self.args = args
        self.teacher = teacher
        self.student = student
        self.train_dataset = train_dataset
        self.collate_fn_teacher = collate_fn_teacher
        self.collate_fn_student = collate_fn_student

        # step 1:3
        self.layer_map: Dict[int, int] = {                        
            1: 3,
            2: 6,
            3: 9,
            4: 12,
            5: 15,
        }
        
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

        self.projection = HiddenProjection(
            self.student.encoder.config.hidden_size, 
            self.teacher.encoder.config.hidden_size)

        self.setup_logging()

        self.lab_to_exclude = None
        
        if getattr(args, 'seed', None):
            set_seed(args.seed)

        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
            
        self.train_dataloader_teacher = self.create_dataloader(train_dataset, collate_fn_teacher, shuffle=False)
        self.train_dataloader_student = self.create_dataloader(train_dataset, collate_fn_student, shuffle=False)
        
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_lr_scheduler()
            
        objs = [self.student,
                self.teacher,
                self.projection,
                self.optimizer,
                self.train_dataloader_teacher,
                self.train_dataloader_student,
                self.lr_scheduler]

        prepared = self.accelerator.prepare(*objs)

        (self.student,
         self.teacher,
         self.projection,
         self.optimizer,
         self.train_dataloader_teacher,
         self.train_dataloader_student,
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

    
    def create_dataloader(self, dataset, collate_fn, shuffle=False):
        if dataset is None:
            return None    

        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=getattr(self.args, 'num_workers', 4),
            pin_memory=True,
            drop_last=shuffle 
        )
        
    def create_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in list(self.student.named_parameters()) +
                                     list(self.projection.named_parameters())
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay
            },
            {
                "params": [
                    p for n, p in list(self.student.named_parameters()) +
                                     list(self.projection.named_parameters())
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
    def create_lr_scheduler(self):
        """Create learning rate scheduler with optional warmup."""
        steps_per_epoch = len(self.train_dataloader_student)
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
            model=self.accelerator.unwrap_model(self.student), 
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
        self.logger.info(f"  Total optimization steps = {len(self.train_dataloader_student) * self.args.num_epochs}")
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
                    self.accelerator.unwrap_model(self.student)
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
                self.student.half()
            
            self.evaluator.refresh_model(
                self.accelerator.unwrap_model(self.student)
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

        self.student.train()
        self.projection.train()
        self.teacher.eval()
        
        epoch_loss = 0.0
        epoch_embd = 0.0
        epoch_hidn = 0.0
        epoch_task = 0.0
        num_batches = 0

        train_dataloader = zip(self.train_dataloader_student, self.train_dataloader_teacher) 
        progress_bar = tqdm(
            train_dataloader,
            total=min(
                len(self.train_dataloader_student),
                len(self.train_dataloader_teacher),
            ),
            desc=f"Training epoch {epoch + 1}",
            disable=not self.accelerator.is_local_main_process,            
        )
        for batch in progress_bar:
            with self.accelerator.accumulate(self.student):
                                
                loss, components = self.combined_loss(batch, self.layer_map)

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients and getattr(self.args, 'max_grad_norm', None):
                    self.accelerator.clip_grad_norm_(
                        self.student.parameters(),
                        self.args.max_grad_norm
                    )

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                gathered_loss = self.accelerator.gather(loss).mean().item()
                gathered_embd = self.accelerator.gather(
                    torch.tensor(components["embd"], device=loss.device)
                ).mean().item()
        
                gathered_hidn = self.accelerator.gather(
                    torch.tensor(components["hidn"], device=loss.device)
                ).mean().item()
        
                gathered_task = self.accelerator.gather(
                    torch.tensor(components["ntxent"], device=loss.device)
                ).mean().item()
        
                epoch_loss += gathered_loss
                epoch_embd += gathered_embd
                epoch_hidn += gathered_hidn
                epoch_task += gathered_task
        
                num_batches += 1
        
                progress_bar.set_postfix({
                    'loss': f"{gathered_loss:.4f}",
                    'embd': f"{gathered_embd:.4f}",
                    'hidn': f"{gathered_hidn:.4f}",
                    'ntxent': f"{gathered_task:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
        
                if self.accelerator.sync_gradients:
                    self.global_step += 1
        
                    if self.global_step % self.args.log_freq == 0:
                        self.accelerator.log({
                            'train/loss_step': gathered_loss,
                            'train/loss_embd': gathered_embd,
                            'train/loss_hidn': gathered_hidn,
                            'train/loss_ntxent': gathered_task,
                            'train/lr': self.optimizer.param_groups[0]['lr'],
                        }, step=self.global_step)

        return {
            'train/loss_epoch': epoch_loss / num_batches,
            'train/loss_embd_epoch': epoch_embd / num_batches,
            'train/loss_hidn_epoch': epoch_hidn / num_batches,
            'train/loss_ntxent_epoch': epoch_task / num_batches,
            'train/epoch': epoch + 1
        }

        
    def ntxent_loss(self, embeddings, in_batch_negative_masks) -> Dict[str, torch.Tensor]:        
        
        anchor_embeddings, positive_embeddings, negative_embeddings = embeddings
        
        loss_output = self.loss_fn(
            anchor_embeddings=anchor_embeddings,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings,
            in_batch_negative_masks=in_batch_negative_masks
        )
        return {"loss": loss_output}


    def hidden_loss(
        self,
        student_hiddens: Tuple[torch.Tensor],
        teacher_hiddens: Tuple[torch.Tensor],
        layer_map: Dict[int, int]
    ) -> torch.Tensor:
    
        loss = torch.tensor(0.0, device=student_hiddens[0].device)
        for s_idx, t_idx in layer_map.items():
            s_h = student_hiddens[s_idx]
            t_h = teacher_hiddens[t_idx]
            projected = self.projection(s_h)
            loss += F.mse_loss(projected, t_h)
        return loss / len(layer_map)

    def embedding_loss(
        self,
        student_embeddings: Tuple[torch.Tensor],
        teacher_embeddings: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        
        s_emb = student_embeddings
        t_emb = teacher_embeddings
        return F.mse_loss(self.projection(s_emb), t_emb)

    
    def combined_loss(
        self,
        batch,        
        layer_map: Dict[int, int],
        alpha: float = 0.7,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        batch_student, batch_teacher = batch
        
        (anchor_s, positive_s, negative_s) = self.gen_embedding(model=self.student, batch=batch_student)
        (anchor_t, positive_t, negative_t) = self.gen_embedding(model=self.teacher, batch=batch_teacher)

        embeddings_s = (anchor_s['embeddings'], positive_s['embeddings'], negative_s['embeddings'])

        l_embd = self.embedding_loss(
            anchor_s['embeddings'],
            anchor_t['embeddings'],
        )
        l_hidn = self.hidden_loss(
            anchor_s['hidden_states'],
            anchor_t['hidden_states'],
            layer_map
        )
        l_task = self.ntxent_loss(embeddings_s, batch_student.get('in_batch_negative_masks'))
    
        total = l_embd + l_hidn +   (1 - alpha) * l_task['loss']
    
        components = {
            "embd": l_embd.item(),
            "hidn": l_hidn.item(),
            "ntxent": l_task['loss'].item(),
        }
        return total, components
    

    def gen_embedding(self, model, batch):
    
        anchor = model(
            batch['anchor_input_ids'],
            batch['anchor_attention_mask'],
            return_dict=True
        )
    
        positive = model(
            batch['positive_input_ids'],
            batch['positive_attention_mask'],
            return_dict=True
        )
    
        negative = None
        if batch.get('negative_input_ids') is not None:
    
            neg_ids = batch['negative_input_ids']
            neg_mask = batch['negative_attention_mask']
    
            batch_size, num_negs, seq_len = neg_ids.shape
    
            neg_ids = neg_ids.view(-1, seq_len)
            neg_mask = neg_mask.view(-1, seq_len)
    
            neg_outputs = model(
                neg_ids,
                neg_mask,
                return_dict=True
            )

            negative = {
                "embeddings": neg_outputs["embeddings"].view(batch_size, num_negs, -1),
    
                "token_embeddings": neg_outputs["token_embeddings"].view(batch_size, num_negs, seq_len, -1)
                if neg_outputs.get("token_embeddings") is not None else None,
    
                "hidden_states": neg_outputs.get("hidden_states"),
                "attentions": neg_outputs.get("attentions"),
            }
    
        return anchor, positive, negative

        
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        if not self.accelerator.is_main_process:
            return
        
        save_path = os.path.join(self.args.checkpoint_dir, self.args.run_name, name)
            
        os.makedirs(save_path, exist_ok=True)
        unwrapped_model = self.accelerator.unwrap_model(self.student)
        unwrapped_model.save_pretrained(save_path)
        unwrapped_model.tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
        
        trainer_state = {
            'epoch': self.global_step // len(self.train_dataloader_student),
            'global_step': self.global_step,
            'best_metric': self.best_score,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'args': vars(self.args),
        }
        
        torch.save(trainer_state, os.path.join(save_path, 'trainer_state.pt'))
        self.logger.info(f"Checkpoint saved to {save_path}")
