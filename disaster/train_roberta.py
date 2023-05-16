from dataclasses import dataclass
from functools import partial
from typing import Optional
import torch
from torch.utils.data import random_split
import torchtext.models
import disaster.data as dd 
from disaster.app import DisasterApp, load_checkpoint

@dataclass
class RobertaConfig:
    dataset: str
    use_all_data: bool
    checkpoint: Optional[str]
    checkpoint_freq: Optional[int]
    checkpoint_dir: str
    checkpoint_name: str
    tensorboard_log_dir: str
    log_freq: int
    batch_size: int
    epochs: int


def train_roberta(config: RobertaConfig) -> None:
    bundle = torchtext.models.ROBERTA_LARGE_ENCODER
    tokenize = bundle.transform()

    if config.checkpoint is not None:
        app = load_checkpoint(config.checkpoint, inference=False, checkpoint_name=config.checkpoint_name, checkpoint_dir=config.checkpoint_dir, tensorboard_log_dir=config.tensorboard_log_dir)
    else:
        head = torchtext.models.RobertaClassificationHead(2, 1024, 1024)
        model = bundle.get_model(head=head, freeze_encoder=True)
        lr = 0.00005
        l2_weight_decay = 0.0025
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_weight_decay)
        app = DisasterApp(model=model,
            optimizer=optimizer,
            checkpoint_name=config.checkpoint_name,
            checkpoint_dir=config.checkpoint_dir,
            tensorboard_log_dir=config.tensorboard_log_dir)

    data = dd.load_csv_dataset(config.dataset)
    data = dd.tokenize_dataset(data, tokenize)

    train_data, val_data = random_split(data, [0.9, 0.1])
    transform = partial(dd.default_transform, seq_len=None, padding_index=1, train=True)
    train_data = dd.build_pipeline(train_data, transform, config.batch_size, shuffle=True)
    val_data = dd.build_pipeline(val_data, transform, config.batch_size, shuffle=True)

    app.train(train_data,
        val_data=val_data,
        num_epochs=config.epochs,
        checkpoint_freq=config.checkpoint_freq,
        log_freq=config.log_freq)
