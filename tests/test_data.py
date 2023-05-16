import unittest
import disaster.data as dd
import torch

class TestCSVLoading(unittest.TestCase):

    def test_load_csv(self):
        csv = [
            "id,keyword,location,text,target",
            "1,,,this is an example tweet,1",
            "4,,,this is also a tweet,0",
        ]

        data = dd._parse_csv(csv)
        self.assertTrue(len(data) == 2)
        self.assertEqual(data[0], (1, "this is an example tweet", 1))
        self.assertEqual(data[1], (4, "this is also a tweet", 0))

class TestDataTransformations(unittest.TestCase):

    def test_padding(self):
        x = [1, 2, 3, 4]
        y = dd.to_length(x, 6, 42)
        self.assertListEqual(y, [1, 2, 3, 4, 42, 42])

        x = [1, 2, 3, 4]
        y = dd.to_length(x, 4, 42)
        self.assertListEqual(y, [1, 2, 3, 4])

    def test_truncate(self):
        x = [1, 2, 3, 4]
        y = dd.to_length(x, 2, 42)
        self.assertListEqual(y, [1, 2])

    def test_transform_batch_train(self):
        batch = [(1, [1, 2, 3], 1), (2, [4, 5, 6, 7], 0)]
        inputs, labels = dd.default_transform(batch, padding_index=12, train=True)
        self.assertIsInstance(inputs, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)

        self.assertTupleEqual(inputs.size(), (2, 4))
        self.assertTupleEqual(labels.size(), (2,))

        self.assertListEqual(inputs.tolist(), [[1, 2, 3, 12], [4, 5, 6, 7]])
        self.assertListEqual(labels.tolist(), [1, 0])

        inputs, _ = dd.default_transform(batch, seq_len=10, train=True)
        self.assertTupleEqual(inputs.size(), (2, 10))

    def test_transform_batch_test(self):
        batch = [(1, [1, 2, 3], None), (2, [4, 5, 6, 7], None)]
        ids, inputs = dd.default_transform(batch, padding_index=12, train=False)
        self.assertIsInstance(inputs, torch.Tensor)
        self.assertIsInstance(ids, list)

        self.assertTupleEqual(inputs.size(), (2, 4))
        self.assertListEqual(inputs.tolist(), [[1, 2, 3, 12], [4, 5, 6, 7]])

        self.assertListEqual(ids, [1, 2])

    def test_transform_single(self):
        x = (1, [1, 2, 3], 1)
        inputs, labels = dd.default_transform(x, seq_len=5, padding_index=12, train=True)
        self.assertIsInstance(inputs, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)

        self.assertTupleEqual(inputs.size(), (5,))
        self.assertTupleEqual(labels.size(), ())

        self.assertListEqual(inputs.tolist(), [1, 2, 3, 12, 12])
        self.assertEqual(labels.item(), 1)