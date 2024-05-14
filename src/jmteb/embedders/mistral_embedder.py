from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from torch import Tensor

from jmteb.embedders.base import TextEmbedder


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


class MistralEmbedder(TextEmbedder):
    """SentenceBERT embedder."""

    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 1,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(
            model_name_or_path, device_map="auto", use_flash_attention_2=True, torch_dtype=torch.bfloat16
            )
        self.batch_size = 1
        self.device = self.model.device
        self.normalize_embeddings = normalize_embeddings
        # OOM回避、評価速度向上のため   
        self.model.max_seq_length = 512
        # prefix取得のため
        self.model_name_or_path = model_name_or_path

    def encode(self, text: str | list[str]) -> np.ndarray:
        return self.get_embedding(text)

    def get_output_dim(self) -> int:
        return 4096

    # https://huggingface.co/intfloat/e5-mistral-7b-instruct#usage
    def get_embedding(self, texts):
        assert type(texts) == list
        with torch.no_grad():
            batch_dict = self.tokenizer(
                texts,
                max_length=512,
                return_attention_mask=False,
                padding=False,
                truncation=True,
            )

            # append eos_token_id to every input_ids
            batch_dict["input_ids"] = [
                input_ids + [self.tokenizer.eos_token_id]
                for input_ids in batch_dict["input_ids"]
            ]
            batch_dict = self.tokenizer.pad(
                batch_dict, padding=True, return_attention_mask=True, return_tensors="pt"
            ).to(self.model.device)
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )

            # normalize embeddings
            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().float()
