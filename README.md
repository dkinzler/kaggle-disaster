# Kaggle NLP with Disaster Tweets

A simple attempt at the [Kaggle NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) challenge.

Two approaches are supported by this repo:
* Training a classification head on top of a pretrained RoBERTa model
* Training a Transformer model based on pretrained word embeddings (e.g. fastText)

For all available commands and options run `python -m disaster --help`.

## Usage

### Using pretrained RoBERTa 

```bash
python -m disaster train_roberta --dataset train.csv --batch_size 32 --epochs 20
```

Every 10 training epochs a checkpoint will be saved to the `ckpts` directory. Edit [train_roberta.py](disaster/train_roberta.py) to change model and optimization parameters like number and size of layers, learning rate or dropout.

To resume training from a previously saved checkpoint run:

```bash
python -m disaster train_roberta --dataset train.csv --checkpoint ckpt.pt
```

### Using pretrained word embeddings

Download [fastText word embeddings](https://fasttext.cc/docs/en/english-vectors.html) and extract a subset of the words.

```bash
python -m disaster extract crawl-300d-2M.vec --num_words 200000
```

This will create two files:
* `token_dict.pt` token dictionary mapping words to indices
* `embeddings.pt`  tensor containing the first 200000 word embeddings

To train a Transformer model run: 

```bash
python -m disaster train --dataset "train.csv" --token_dict "token_dict.pt" --pretrained_embeddings "embeddings.pt" --batch_size 32 --epochs 30
```

Edit [train_transformer.py](disaster/train_transformer.py) to change model and optimization parameters like number of layers, attention heads, learning rate or dropout.

### Test and Inference

To create predictions for the test dataset using a model checkpoint run:

```bash
python -m disaster test ckpts/model-10.pt --dataset test.csv --token_dict "token_dict.pt"
```

This will create a file `result.csv` that can be submitted for the Kaggle challenge.

To interactively classify sentences from the command line run:

```bash
python -m disaster inference ckpts/model-10.pt --token_dict "token_dict.pt"
```

## License

[MIT](LICENSE)