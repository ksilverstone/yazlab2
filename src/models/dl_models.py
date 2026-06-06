import torch
import torch.nn as nn
import torch.nn.functional as F


class AnomalyLSTM(nn.Module):
    """Zaman serilerinde anomali tespiti için LSTM tabanlı Derin Öğrenme mimarisi."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super(AnomalyLSTM, self).__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)
        return out


class AnomalyGRU(nn.Module):
    """Zaman serilerinde anomali tespiti için GRU tabanlı Derin Öğrenme mimarisi."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super(AnomalyGRU, self).__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)
        return out


class AnomalyCNN1D(nn.Module):
    """Zaman serilerinde anomali tespiti için 1D-CNN mimarisi."""

    def __init__(
        self, num_features: int, hidden_channels: int, kernel_size: int, dropout: float = 0.0
    ):
        super(AnomalyCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels * 2,
            kernel_size=kernel_size,
            padding="same",
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels * 2, 2)

    def forward(self, x):
        # (batch, window, features) -> (batch, features, window) for Conv1d
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        # Global Average Pooling
        x = torch.mean(x, dim=2)
        out = self.fc(x)
        return out


def build_dl_model(model_name: str, input_size: int, config: dict):
    """Config'den model adına göre DL modeli oluşturur."""
    models_config = config.get("deep_learning", {}).get("models", {})

    if model_name == "lstm":
        mc = models_config.get("lstm", {})
        return AnomalyLSTM(
            input_size=input_size,
            hidden_size=mc.get("hidden_size", 64),
            num_layers=mc.get("num_layers", 2),
            dropout=mc.get("dropout", 0.2),
        )
    elif model_name == "gru":
        mc = models_config.get("gru", {})
        return AnomalyGRU(
            input_size=input_size,
            hidden_size=mc.get("hidden_size", 64),
            num_layers=mc.get("num_layers", 2),
            dropout=mc.get("dropout", 0.2),
        )
    elif model_name == "cnn_1d":
        mc = models_config.get("cnn_1d", {})
        return AnomalyCNN1D(
            num_features=input_size,
            hidden_channels=mc.get("out_channels", 64),
            kernel_size=mc.get("kernel_size", 3),
            dropout=mc.get("dropout", 0.2),
        )
    else:
        raise ValueError(f"Bilinmeyen model adı: {model_name}")
