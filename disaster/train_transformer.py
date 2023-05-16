import contextlib
from dataclasses import dataclass
from functools import partial
from typing import Optional
import torch
from torch.utils.data import random_split
from disaster.models import TransformerModelConfig, TransformerModel
from disaster.app import DisasterApp, load_checkpoint
import disaster.data as dd 

@dataclass
class OptimizerConfig:
    lr: float
    l2_reg: float

@dataclass
class TrainConfig:
    dataset: str
    token_dict: str
    pretrained_embeddings: str
    use_all_data: bool
    checkpoint: Optional[str]
    checkpoint_freq: Optional[int]
    checkpoint_dir: str
    checkpoint_name: str
    tensorboard_log_dir: str
    profile: bool
    log_freq: int
    batch_size: int
    epochs: int

def train(config: TrainConfig) -> None:
    data = dd.load_csv_dataset(config.dataset)

    tokenizer = dd.BasicTokenizer(config.token_dict)
    data = dd.tokenize_dataset(data, tokenizer.tokenize)

    seq_len = max(len(y[1]) for y in data)
    if seq_len < 100:
        seq_len = 100


    app = None
    if config.checkpoint is not None:
        app = load_checkpoint(checkpoint=config.checkpoint, checkpoint_dir=config.checkpoint_dir, checkpoint_name=config.checkpoint_name, tensorboard_log_dir=config.tensorboard_log_dir)
    else:
        model_config = TransformerModelConfig(
            seq_len=seq_len,
            pretrained_embeddings=config.pretrained_embeddings,
            ff_dim=512,
            linear_dim=1024,
            embedding_dropout=0.4,
            transformer_dropout=0.4,
            linear_dropout=0.4,
        )
        model = TransformerModel(model_config)

        optimizer_config = OptimizerConfig(
            lr= 0.00005,
            l2_reg=0.0025,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.l2_reg)

        app = DisasterApp(model=model, optimizer=optimizer, checkpoint_dir=config.checkpoint_dir, checkpoint_name=config.checkpoint_name, tensorboard_log_dir=config.tensorboard_log_dir)

    data_transform = partial(dd.default_transform, seq_len=seq_len, padding_index=0, train=True)

    if config.use_all_data:
        train_data, val_data = data, None
    else:
        train_data, val_data = random_split(data, [0.9, 0.1])
        val_data = dd.build_pipeline(val_data, data_transform, batch_size=config.batch_size, shuffle=True)
    train_data = dd.build_pipeline(train_data, data_transform, batch_size=config.batch_size, shuffle=True)

    ctx_mgr = build_profiler() if config.profile else contextlib.nullcontext()
    with ctx_mgr as profiler:
        app.train(train_data, val_data=val_data, num_epochs=config.epochs, log_freq=config.log_freq, checkpoint_freq=config.checkpoint_freq, profiler=profiler)

def build_profiler() -> torch.profiler.profile:
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("logs"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )