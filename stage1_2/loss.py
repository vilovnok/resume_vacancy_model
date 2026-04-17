import torch
import torch.nn as nn
from typing import List, Dict, Set, Tuple, Optional, Any
import torch.nn.functional as F



class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        passage_embeddings: torch.Tensor,
        labels: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
        in_batch_negatives: bool = None,
        in_batch_negative_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: 
        
        cosine_sim = F.cosine_similarity(query_embeddings, passage_embeddings)
        return self.mse(cosine_sim, labels)


class MarginLoss(nn.Module):   
    def __init__(self, args):
        super(MarginLoss, self).__init__()
        
        self.margin = args.margin
        self.reduction = args.reduction
        self.hard_negative_strategy = args.hard_negative_strategy

    
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
        in_batch_negatives: bool = None,
        in_batch_negative_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
 
        batch_size = anchor_embeddings.size(0)
        device = anchor_embeddings.device
        
        if in_batch_negatives is None:
            in_batch_strategies = [
                "in_batch", 
                "mixed_batch_hard", 
                "mixed_batch_esco", 
                "mixed_all"
            ]
            in_batch_negatives = self.hard_negative_strategy in in_batch_strategies
            
        anchor_embeddings = F.normalize(anchor_embeddings, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, dim=1)
        if negative_embeddings is not None:
            negative_embeddings = F.normalize(negative_embeddings, dim=2)
        
        positive_similarities = F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=1)
        
        all_negative_similarities = []
        
        if in_batch_negatives:
            batch_similarities = torch.matmul(anchor_embeddings, positive_embeddings.T)
            
            if in_batch_negative_masks is not None:
                batch_negative_similarities = batch_similarities.masked_fill(
                    ~in_batch_negative_masks, float('-inf')
                )
            else:
                # Default: mask out positive pairs (diagonal)
                mask = torch.eye(batch_size, device=device, dtype=torch.bool)
                batch_negative_similarities = batch_similarities.masked_fill(mask, float('-inf'))
                
            all_negative_similarities.append(batch_negative_similarities)    
            
        if negative_embeddings is not None:
            hard_negative_similarities = torch.bmm(
                anchor_embeddings.unsqueeze(1), 
                negative_embeddings.transpose(1, 2)
            ).squeeze(1)  
    
            all_negative_similarities.append(hard_negative_similarities)
        
        if all_negative_similarities:
            negative_similarities = torch.cat(all_negative_similarities, dim=1)
        else:
            raise ValueError("No negatives provided (either in_batch_negatives=True or negative_embeddings must be provided)")
        
        valid_negatives_mask = negative_similarities != float('-inf')
        
        positive_similarities_expanded = positive_similarities.unsqueeze(1)
        margin_losses = torch.clamp(
            self.margin - positive_similarities_expanded + negative_similarities, 
            min=0.0
        )
        
        if valid_negatives_mask.any(dim=1).all():
            loss_per_sample = (margin_losses * valid_negatives_mask.float()).sum(dim=1) / valid_negatives_mask.float().sum(dim=1)
        else:
            loss_per_sample = margin_losses.mean(dim=1)
        
        if self.reduction == "mean":
            loss = loss_per_sample.mean()
        elif self.reduction == "sum":
            loss = loss_per_sample.sum()
        else:
            loss = loss_per_sample

        return loss




class NTXent(nn.Module):
    def __init__(self, args):
        super(NTXent, self).__init__()
        self.reduction = args.reduction
        self.hard_negative_strategy = args.hard_negative_strategy
        
        if args.pre_train:
            self.temperature = args.temperature_pre_train
        else:
            self.temperature = args.temperature_main
        
    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
        in_batch_negatives: bool = None,
        in_batch_negative_masks: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
 
        batch_size = anchor_embeddings.size(0)
        device = anchor_embeddings.device
        
        if in_batch_negatives is None:
            in_batch_strategies = [
                "in_batch", 
                "mixed_batch_hard", 
                "mixed_batch_esco", 
                "mixed_all"
            ]
            in_batch_negatives = self.hard_negative_strategy in in_batch_strategies
        
        anchor_embeddings = F.normalize(anchor_embeddings, dim=1)
        positive_embeddings = F.normalize(positive_embeddings, dim=1)
        if negative_embeddings is not None:
            negative_embeddings = F.normalize(negative_embeddings, dim=2)
            
        positive_similarities = F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=1)
        positive_similarities = positive_similarities / self.temperature 
        
        all_negative_similarities = []
        if in_batch_negatives:
            batch_similarities = torch.matmul(anchor_embeddings, positive_embeddings.T)
            batch_similarities = batch_similarities / self.temperature
            
            if in_batch_negative_masks is not None:
                batch_negative_similarities = batch_similarities.masked_fill(
                    ~in_batch_negative_masks, float('-inf')
                )
            else:
                mask = torch.eye(batch_size, device=device, dtype=torch.bool)
                batch_negative_similarities = batch_similarities.masked_fill(mask, float('-inf'))
                
            all_negative_similarities.append(batch_negative_similarities)
            
       
        if negative_embeddings is not None:
            
            hard_negative_similarities = torch.bmm(
                anchor_embeddings.unsqueeze(1), 
                negative_embeddings.transpose(1, 2)
            ).squeeze(1)
            
            hard_negative_similarities = hard_negative_similarities / self.temperature
            all_negative_similarities.append(hard_negative_similarities)
            
        if all_negative_similarities:
            negative_similarities = torch.cat(all_negative_similarities, dim=1)
        else:
            raise ValueError("No negatives provided (either in_batch_negatives=True or negative_embeddings must be provided)")
        
        # compute loss
        numerator = torch.exp(positive_similarities) 
        all_similarities = torch.cat([
            positive_similarities.unsqueeze(1),
            negative_similarities
        ], dim=1)
        
        denominator = torch.sum(torch.exp(all_similarities), dim=1)

        loss = -torch.log(numerator / denominator)
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
    
        return loss


class SymmetricNTXent(nn.Module):
    def __init__(self, base_loss: NTXent):
        super().__init__()
        self.base_loss = base_loss         

    @torch.no_grad()
    def _transpose_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None if mask is None else mask.T

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        *,
        negative_embeddings: Optional[torch.Tensor] = None,
        in_batch_negative_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        loss_ab = self.base_loss(
            anchor_embeddings        = anchor_embeddings,
            positive_embeddings      = positive_embeddings,
            negative_embeddings      = negative_embeddings,
            in_batch_negative_masks  = in_batch_negative_masks,
        )

        loss_ba = self.base_loss(
            anchor_embeddings        = positive_embeddings,      
            positive_embeddings      = anchor_embeddings,
            negative_embeddings      = negative_embeddings,      
            in_batch_negative_masks  = self._transpose_mask(in_batch_negative_masks),
        )

        return 0.5 * (loss_ab + loss_ba)


class ContrastiveLossWrapper(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        
        loss_type = args.loss_type.lower()
        negative_sampling = args.hard_negative_strategy
        
        if loss_type == "cos_sim":
            self.loss_fn = CosineSimilarityLoss(args)
        elif loss_type == "ntxent" and negative_sampling == 'in_batch':
            base_loss = NTXent(args)
            self.loss_fn = SymmetricNTXent(base_loss)
        elif loss_type == "margin":
            self.loss_fn = MarginLoss(args)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, *args, **kwargs):
        return self.loss_fn(*args, **kwargs)