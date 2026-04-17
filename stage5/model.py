import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import os
import json



class BiEncoderWrapper(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.model_name = args.model_name
        self.max_length = getattr(args, "max_length", 512)
        self.dropout_rate = getattr(args, "dropout_rate", 0.1)
        self.pooling_strategy = getattr(args, "pooling_strategy", "mean")

        self.logger = logging.getLogger(__name__)

        self.logger.info(
            "Loading pre-trained model: %s", self.model_name
        )

        self.encoder = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.hidden_size = self.encoder.config.hidden_size
        self.output_dim = self.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            
        )

        token_embeddings = encoder_outputs.last_hidden_state
        hidden_states = encoder_outputs.hidden_states
        
        sentence_embeddings = self._pool_embeddings(
            token_embeddings,
            attention_mask,
        )

        normalized_embeddings = F.normalize(sentence_embeddings, dim=1)

        if return_dict:
            return {
                "embeddings": normalized_embeddings,
                "pooled_embeddings": sentence_embeddings,
                "token_embeddings": token_embeddings,
                "hidden_states": hidden_states, 
            }
        return normalized_embeddings



    
    def _pool_embeddings(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        if self.pooling_strategy == "mean":
            mask = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()
            ).float()

            sum_embeddings = torch.sum(
                token_embeddings * mask,
                dim=1,
            )

            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

            return sum_embeddings / sum_mask
            
        raise ValueError(
            f"Unknown pooling strategy: {self.pooling_strategy}"
        )

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 64,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = True,
        device: Optional[torch.device] = None,
        normalize_embeddings: bool = False,
        return_dict=True
    ) -> Union[torch.Tensor, np.ndarray]:

        if device is None:
            device = next(self.parameters()).device

        self.eval()
        all_embeddings = []

        total_batches = math.ceil(len(sentences) / batch_size)
        iterator = range(0, len(sentences), batch_size)

        if show_progress_bar:
            iterator = tqdm(
                iterator,
                desc="Encoding Sentences",
                total=total_batches,
            )

        with torch.no_grad():
            for start_idx in iterator:
                batch_sentences = sentences[
                    start_idx : start_idx + batch_size
                ]

                encoded = self.tokenizer(
                    batch_sentences,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                encoded = {
                    k: v.to(device)
                    for k, v in encoded.items()
                }

                outputs = self.forward(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    return_dict=return_dict
                )

                embeddings = outputs['embeddings']
                pooled_embeddings = outputs['pooled_embeddings']
                token_embeddings = outputs['token_embeddings']
                hidden_states = outputs['hidden_states']

                if normalize_embeddings:
                    embeddings = F.normalize(
                        outputs['embeddings'],
                        p=2,
                        dim=1,
                    )

                all_embeddings.append(embeddings.cpu())
        all_embeddings = torch.cat(all_embeddings, dim=0)

        if convert_to_numpy:
            return {
                "embeddings": all_embeddings.numpy().astype("float32"),
                "pooled_embeddings": pooled_embeddings,
                "token_embeddings": token_embeddings,
                "hidden_states": hidden_states,
            }

        return {
                "embeddings": all_embeddings,
                "pooled_embeddings": pooled_embeddings,
                "token_embeddings": token_embeddings,
                "hidden_states": hidden_states,
            }


    
    def get_embedding_dimension(self) -> int:
        return self.output_dim

    def save_pretrained(self, save_path: str) -> None:
        os.makedirs(save_path, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "model_name": self.model_name,
                "dropout_rate": self.dropout_rate,
                "pooling_strategy": self.pooling_strategy,
                "output_dim": self.output_dim,
                "hidden_size": self.hidden_size,
                "max_length": self.max_length,
            },
            os.path.join(save_path, "pytorch_model.bin"),
        )

        config = {
            "model_name": self.model_name,
            "dropout_rate": self.dropout_rate,
            "pooling_strategy": self.pooling_strategy,
            "output_dim": self.output_dim,
            "hidden_size": self.hidden_size,
            "max_length": self.max_length,
        }

        with open(
            os.path.join(save_path, "config.json"),
            "w",
        ) as file:
            json.dump(config, file, indent=2)

        self.logger.info("Model saved to %s", save_path)

    @classmethod
    def from_pretrained(
        cls,
        load_path: str,
        args=None,
    ) -> "BiEncoderModel":

        with open(
            os.path.join(load_path, "config.json"),
            "r",
        ) as file:
            config = json.load(file)

        if args is None:

            class Args:
                pass

            args = Args()
            for key, value in config.items():
                setattr(args, key, value)

        model = cls(args)

        checkpoint = torch.load(
            os.path.join(load_path, "pytorch_model.bin"),
            map_location="cpu",
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        return model


class HiddenProjection(nn.Module):
    def __init__(self, student_size: int = 348, teacher_size: int = 768):
        super().__init__()
        self.proj = nn.Linear(student_size, teacher_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)