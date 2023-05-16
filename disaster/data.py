"""Functions to build data pipelines for the Kaggle NLP with Disaster Tweets challenge."""

import csv
from typing import Callable, Iterable, Optional
import torch
import torchtext.data
from torchdata.datapipes.map import MapDataPipe, SequenceWrapper
from torchdata.datapipes.iter import IterDataPipe


def load_csv_dataset(filepath: str) -> list[tuple[int, str, Optional[int]]]:
    """Load a dataset file for the Kaggle NLP with Disaster Tweets challenge.
    Result will be a list of tuples, each representing a sample and containing the sample id (int),
    tweet text (str) and label (int or None).
    A 1 (0) label indicates that the tweet is (not) about a disaster event.
    When loading the challenge test set, the labels will be None.
    """
    with open(filepath, mode='r', newline='', encoding="utf8") as file:
        return _parse_csv(file)

def _parse_csv(f: Iterable[str]):
    data = []
    reader = csv.DictReader(f)
    for row in reader:
        # row is a dict with keys id, keyword, location, text, target (not for test dataset)
        text = row["text"]

        label=None
        if "target" in row:
            label = int(row["target"])

        sample_id = int(row["id"])

        data.append((sample_id, text, label))

    return data

def tokenize_dataset(data: list[tuple[int, str, Optional[int]]], tokenize: Callable) -> list[tuple[int, list[int], Optional[int]]]:
    """Tokenize the text of each sample of a dataset.
    Since the dataset for the Kaggle challenge is small, we can just tokenize all
    sentences once at the beginning.
    """
    return [(id, tokenize(text), label) for id, text, label in data]

class BasicTokenizer:
    """BasicTokenizer loads its token dictionary from a file and uses the basic_english tokenizer
    from the torchtext.data package. This will lowercase the text and split it into words.
    Note however that e.g. fastText word embeddings are case-sensitive. There are separate
    embeddings for "and" and "And", so some information will be lost.
    """
    def __init__(self, token_dict_file: str) -> None:
        self._token_dict = torch.load(token_dict_file)
        self._tokenizer = torchtext.data.get_tokenizer("basic_english")

    def tokenize(self, text: str | list[str], unknown_index=1) -> list[int] | list[list[int]]:
        """Split text into tokens and map them to token indices.
        Supports single elements (str) or batches (list[str]).
        """
        if isinstance(text, list):
            return [self._tokenize_single(x, unknown_index=unknown_index) for x in text]
        else:
            return self._tokenize_single(text, unknown_index=unknown_index)
    
    def _tokenize_single(self, text: str, unknown_index=1) -> list[int]:
        return [self._token_dict.get(y, unknown_index) for y in self._tokenizer(text)]


def build_pipeline(data, transform: Optional[Callable], batch_size: Optional[int], shuffle=True) -> IterDataPipe | MapDataPipe:
    """Returns a pipeline that applies the given transform function to each element/batch.
    If batch_size is None, no batching is performed.
    """
    pipeline = SequenceWrapper(data)
    if shuffle:
        pipeline = pipeline.shuffle()
    if batch_size is not None:
        pipeline = pipeline.batch(batch_size)
    if transform is not None:
        pipeline = pipeline.map(transform)
    return pipeline

def default_transform(samples, seq_len=None, padding_index=0, train=True):
    """Transform the given sample or batch of samples to tensors.
    It is assumed that text is already tokenized. The list of token indices is padded to
    the given length (max length of the batch if seq_len is None).
    If train is true returns tokens and labels, otherwise sample ids and tokens.
    """
    if isinstance(samples, list):
        sample_id, tokens, label = [x[0] for x in samples], [x[1] for x in samples], [x[2] for x in samples]
        if seq_len is None:
            seq_len = max(len(x) for x in tokens)
        tokens = [to_length(x, seq_len, padding_index) for x in tokens]
    else:
        sample_id, tokens, label = samples
        if seq_len is not None:
            tokens = to_length(tokens, seq_len, padding_index)

    if train:
        return torch.tensor(tokens), torch.tensor(label)
    else:
        return sample_id, torch.tensor(tokens)


def to_length(seq: list[int], length: int, padding_index) -> list[int]:
    """Pad or truncate sequence to given length."""
    if len(seq) > length:
        return seq[:length]
    if len(seq) < length:
        seq.extend([padding_index for i in range(length-len(seq))])
    return seq