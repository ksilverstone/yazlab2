import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class TimeSeriesDataset(Dataset):
    """Zaman serisi verisi için Sliding Window (Kayan Pencere) uygulayan PyTorch Dataset sınıfı."""
    
    def __init__(self, X, y, window_size: int):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
            
        # Gelen veriyi içeride PyTorch Tensor'lerine (float32 ve long) çevir
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.window_size = window_size

    def __len__(self):
        # Pencere boyutundan dolayı kayıp olan kısımları hesaba katar
        return max(0, len(self.X) - self.window_size + 1)

    def __getitem__(self, idx):
        x_window = self.X[idx : idx + self.window_size]
        # O pencerenin son adımındaki etiketi (hedef tensör) döner
        y_label = self.y[idx + self.window_size - 1]
        
        return x_window, y_label

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, window_size: int, batch_size: int):
    """Train, Validation ve Test setleri için DataLoader nesnelerini oluşturur."""
    
    train_dataset = TimeSeriesDataset(X_train, y_train, window_size)
    val_dataset = TimeSeriesDataset(X_val, y_val, window_size)
    test_dataset = TimeSeriesDataset(X_test, y_test, window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
