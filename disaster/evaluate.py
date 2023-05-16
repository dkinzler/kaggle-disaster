import csv
from dataclasses import dataclass
from functools import partial
import torch
from disaster.app import load_checkpoint
import disaster.data as dd

@dataclass
class TestConfig:
    dataset: str
    token_dict: str
    checkpoint: str
    output: str
    batch_size: int = 32

def test(config: TestConfig) -> None:
    """ Predict labels for the given test dataset using the provided model checkpoint.
    Output is written to the specified file.
    """
    app = load_checkpoint(config.checkpoint, inference=True)
    print(f"loaded checkpoint: {config.checkpoint}")
    print(f"model was trained for {app.epoch} epochs / {app.global_step} steps")
    seq_len = app.model.seq_len

    tokenizer = dd.BasicTokenizer(config.token_dict)
    test_data = dd.load_csv_dataset(config.dataset)
    test_data = dd.tokenize_dataset(test_data, tokenizer.tokenize)
    transform = partial(dd.default_transform, seq_len=seq_len, train=False)
    test_data = dd.build_pipeline(test_data,transform, batch_size=config.batch_size)

    id_to_pred = app.test(test_data)
    print(f"predicted labels for {len(id_to_pred)} samples")
    test_predictions = [(int(id), pred) for id, pred in id_to_pred.items()]
    # sort by id
    test_predictions.sort(key=lambda x: x[0])

    print(f"writing output to {config.output}")
    with open(config.output, mode='w', newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "target"])
        writer.writerows(test_predictions)

@dataclass
class InferenceConfig:
    token_dict: str
    checkpoint: str

def inference(config: InferenceConfig) -> None:
    """ Load the given model checkpoint and predict labels for
    input sentences from the command line.
    """
    app = load_checkpoint(config.checkpoint, inference=True)
    print(f"loading checkpoint: {config.checkpoint}")
    print(f"model was trained for {app.epoch} epochs / {app.global_step} steps")
    seq_len = app.model.seq_len

    tokenizer = dd.BasicTokenizer(config.token_dict)

    while True:
        text = input("Enter query or \"q\" to quit: ").strip()
        if text == "q":
            break

        tokens = torch.tensor(dd.to_length(tokenizer.tokenize(text), length=seq_len, padding_index=0))

        prediction = app.inference(tokens)
        print(f"Predicted class: {prediction}")