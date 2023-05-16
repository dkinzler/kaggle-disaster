from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class BaseModelConfig:
    seq_len: Optional[int]
    num_classes: int = 2
    pretrained_embeddings: Optional[str] = None
    num_embeddings: Optional[int] = None
    embedding_dim: Optional[int] = None

class BaseModel(nn.Module):
    """Base model class that creates embedding layer from pretrained embeddings."""

    def __init__(self, config: BaseModelConfig) -> None:
        super().__init__()
        self.seq_len = config.seq_len
        self.num_classes = config.num_classes

        if config.pretrained_embeddings is None:
            if config.num_embeddings is None:
                raise ValueError("provide number of embeddings")
            elif config.embedding_dim is None:
                raise ValueError("need to provide embedding dimension")

            self.embedding = nn.Embedding(config.num_embeddings, config.embedding_dim, padding_idx = 0)
        else:
            e = torch.load(config.pretrained_embeddings)
            self.embedding = nn.Embedding.from_pretrained(e, freeze=True, padding_idx=0)

        self.embedding_dim = self.embedding.embedding_dim
               
@dataclass
class LinearModelConfig(BaseModelConfig):
    hidden_size: int = 2048
    dropout: float = 0.4

class LinearModel(BaseModel):
    """A simple network with a single hidden dense layer.
    Inputs need to be padded/truncated to have a fixed sequence length, because
    the embeddings of the input sequence are flattened into a single vector.
    """
    def __init__(self, config: LinearModelConfig) -> None:
        super().__init__(config)
        self.embedding_dropout = nn.Dropout1d(config.dropout)
        self.hidden = nn.Linear(self.seq_len*self.embedding_dim, config.hidden_dim)
        self.hidden_dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.hidden_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_batch = x.dim() == 2

        y = self.embedding(x)
        # 1d dropout uses channels-first format, so we need to transpose
        dim1, dim2 = (1, 2) if is_batch else (0, 1)
        y = self.embedding_dropout(torch.transpose(y, dim1, dim2))
        y = torch.transpose(y, dim1, dim2)

        # flatten embeddings into a single vector, need to handle batched and non-batched input differently
        if is_batch:
            y = torch.flatten(y, start_dim=1)
        else:
            y = torch.flatten(y)

        y = F.relu(self.hidden(y))
        y = self.hidden_dropout(y)
        return self.output(y)
        
@dataclass
class TransformerModelConfig(BaseModelConfig):
    num_heads: int = 6
    num_layers: int = 2
    ff_dim: int = 512
    linear_dim: int = 1024
    embedding_dropout: float = 0.4
    transformer_dropout: float = 0.4
    linear_dropout: float = 0.4

class TransformerModel(BaseModel):
    """Model consisting of a transformer encoder and classification head on top.
    Inputs need to be padded/truncated to have a fixed sequence length, because
    the output sequence of the transformer layer is flattened into a single vector.
    """
    def __init__(self, config: TransformerModelConfig) -> None:
        super().__init__(config)
        self.embedding_dropout = nn.Dropout1d(config.embedding_dropout)

        encoder_layer = nn.TransformerEncoderLayer(self.embedding_dim, nhead=config.num_heads, dim_feedforward=config.ff_dim, activation="relu", batch_first=True, dropout=config.transformer_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.linear = nn.Linear(self.seq_len * self.embedding_dim, config.linear_dim)
        self.linear_dropout = nn.Dropout(config.linear_dropout)
        self.output = nn.Linear(config.linear_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        is_batch = x.dim() == 2 

        y = self.embedding(x)
        # 1d dropout uses channels-first format, so we need to transpose
        dim1, dim2 = (1, 2) if is_batch else (0, 1)
        y = self.embedding_dropout(torch.transpose(y, dim1, dim2))
        y = torch.transpose(y, dim1, dim2)

        y = self.transformer_encoder(y)

        # flatten embeddings into a single vector, need to handle batched and non-batched input differently
        if is_batch:
            y = torch.flatten(y, start_dim=1)
        else:
            y = torch.flatten(y)

        y = F.relu(self.linear(y))
        y = self.linear_dropout(y)
        return self.output(y)