import os
import unittest
import torch
from torch import nn
from disaster.app import DisasterApp, load_checkpoint

class TestApp(unittest.TestCase):

    def test_train(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 2),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        app = DisasterApp(model, optimizer)

        train_data = [(torch.rand(4, 10), torch.randint(0, 2, (4,))) for i in range(2)]
        val_data = [(torch.rand(4, 10), torch.randint(0, 2, (4,))) for i in range(2)]

        app.train(train_data, val_data=val_data, num_epochs=2, checkpoint_freq=None)

        self.assertEqual(app.epoch, 2)
        self.assertEqual(app.global_step, 4)

class TestCheckpointing(unittest.TestCase):

    def test_save_and_load_checkpoints(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 2),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        app = DisasterApp(model, optimizer,
            checkpoint_dir=".",
            checkpoint_name="model-test")

        train_data = [(torch.rand(4, 10), torch.randint(0, 2, (4,))) for i in range(2)]
        val_data = [(torch.rand(4, 10), torch.randint(0, 2, (4,))) for i in range(2)]

        app.train(train_data,
            val_data=val_data,
            num_epochs=4,
            checkpoint_freq=2)

        p1 = os.path.join(".", "model-test-1.pt")
        p2 = os.path.join(".", "model-test-3.pt")
        self.assertTrue(os.path.exists(p1))
        self.assertTrue(os.path.exists(p2))

        # reload model
        app = load_checkpoint("model-test-3.pt")
        app.train(train_data,
            val_data=val_data,
            num_epochs=4,
            checkpoint_freq=None)

    def tearDown(self):
        p1 = os.path.join(".", "model-test-1.pt")
        p2 = os.path.join(".", "model-test-3.pt")
        os.remove(p1)
        os.remove(p2)
