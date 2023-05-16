from dataclasses import dataclass
import torch

@dataclass
class ExtractEmbeddingsConfig:
    input_file: str
    num_words: int
    output_embeddings: str
    output_token_dict: str

def extract_embeddings(config: ExtractEmbeddingsConfig) -> None:
    """Extract word vectors from a fastText file and
    output a token dictionary and embedding file.
    """
    with open(config.input_file, mode="r", encoding="utf8") as f:
        total_num_words, dim = map(int, f.readline().split(" ")) 
        num_words = min(config.num_words, total_num_words)
        print(f"extracting {num_words} word embeddings from {config.input_file}")

        t = torch.zeros((num_words+2, dim))
        # index 0 is padding token
        # index 1 is unknown, initialize it randomly
        t[1] = torch.rand((dim,))

        token_dict = {}
        for i in range(num_words):
            split = f.readline().rstrip().split(" ")
            word, vector = split[0], split[1:]
            token_dict[word] = i+2
            for j in range(dim):
                t[i+2, j] = float(vector[j])
                    
            if (i+1) % 10000 == 0:
                print(f"{i+1} words read")

    torch.save(t, config.output_embeddings)
    torch.save(token_dict, config.output_token_dict)
    print(f"saved word embeddings: {config.output_embeddings}")
    print(f"saved token dictionary: {config.output_token_dict}")

