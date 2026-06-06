import unittest
import torch
from src.models.dl_models import build_dl_model

class TestDLModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 5
        self.input_size = 3
        # Dummy Input: (batch, seq_len, input_size)
        self.dummy_input = torch.randn(self.batch_size, self.seq_len, self.input_size)
        
        # Test için geçici config
        self.config = {
            "deep_learning": {
                "lstm_hidden": 16,
                "lstm_layers": 1,
                "gru_hidden": 16,
                "gru_layers": 1,
                "cnn_filters": 16,
                "cnn_kernel": 2,
                "dropout": 0.1
            }
        }

    def test_lstm_forward(self):
        """LSTM modelinin ileri yayılımının çökmeden çalıştığı test edilir."""
        model = build_dl_model("lstm", input_size=self.input_size, config=self.config)
        output = model(self.dummy_input)
        # Binary Classification output shape: (batch_size, 2)
        self.assertEqual(output.shape, (self.batch_size, 2))

    def test_gru_forward(self):
        """GRU modelinin ileri yayılımının çökmeden çalıştığı test edilir."""
        model = build_dl_model("gru", input_size=self.input_size, config=self.config)
        output = model(self.dummy_input)
        self.assertEqual(output.shape, (self.batch_size, 2))

    def test_cnn_forward(self):
        """CNN_1D modelinin ileri yayılımının çökmeden çalıştığı test edilir."""
        model = build_dl_model("cnn_1d", input_size=self.input_size, config=self.config)
        output = model(self.dummy_input)
        self.assertEqual(output.shape, (self.batch_size, 2))

if __name__ == '__main__':
    unittest.main()
