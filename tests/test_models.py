import unittest
import disaster.models as m
import torch

class TestModels(unittest.TestCase):

    def test_transformer_model(self):
        config = m.TransformerModelConfig(
            seq_len=16,
            num_classes=2,
            num_embeddings=32,
            embedding_dim=64,
            num_heads=2,
            num_layers=2,
            ff_dim=64,
            linear_dim=64,
        )
        model = m.TransformerModel(config)

        inputs = torch.tensor([[j for j in range(16)] for i in range(4)])
        outputs = model(inputs)
        self.assertTupleEqual(outputs.size(), (4,2))
