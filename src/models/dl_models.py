import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyLSTM(nn.Module):
    """
    Zaman serilerinde anomali tespiti için LSTM tabanlı Derin Öğrenme mimarisi.
    Kayan pencere (Sliding Window) girdisini alıp iki sınıflı (Normal/Anomali) çıktı üretir.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super(AnomalyLSTM, self).__init__()
        
        # nn.LSTM'de num_layers 1 ise dropout uygulanması hata verebilir
        lstm_dropout = dropout if num_layers > 1 else 0.0
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        # Girdi boyutu: (batch_size, window_size, input_size)
        out, _ = self.lstm(x)
        
        # Yalnızca son zaman adımının (last hidden state) çıktısını alıyoruz
        # out[:, -1, :] -> Boyut: (batch_size, hidden_size)
        last_hidden = out[:, -1, :]
        
        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)
        
        return out


class AnomalyCNN1D(nn.Module):
    """
    Zaman serilerinde anomali tespiti için 1D Evrişimli Sinir Ağı (1D-CNN) mimarisi.
    Veriyi (batch, window, features) formatından (batch, channels, length) formatına çevirerek işler.
    """
    def __init__(self, num_features: int, hidden_channels: int, kernel_size: int, dropout: float = 0.0):
        super(AnomalyCNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels=num_features, 
            out_channels=hidden_channels, 
            kernel_size=kernel_size,
            padding="same"
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels, 
            out_channels=hidden_channels * 2, 
            kernel_size=kernel_size,
            padding="same"
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Çıktı boyutu CrossEntropyLoss için 2 (Normal ve Anomali)
        self.fc = nn.Linear(hidden_channels * 2, 2)

    def forward(self, x):
        # Dataloader veriyi (batch_size, window_size, features) formatında verir.
        # Conv1d ise (batch_size, in_channels, length) bekler. 
        # features boyutunu in_channels olarak alabilmek için permute ediyoruz.
        x = x.permute(0, 2, 1)
        
        # 1. Konvolüsyon + Aktivasyon + Dropout
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        
        # 2. Konvolüsyon + Aktivasyon
        x = F.relu(self.conv2(x))
        
        # Window (length) boyutu kaybolmasın veya uyumsuzluk olmasın diye 
        # Global Average Pooling (Zaman ekseninde ortalama) alarak düzleştiriyoruz.
        # (batch_size, hidden_channels * 2, length) -> (batch_size, hidden_channels * 2)
        x = torch.mean(x, dim=2)
        
        # Tam bağlı katmandan (Linear) geçir
        out = self.fc(x)
        
        return out
