import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TimeSeriesDataset(Dataset):
    """Zaman serisi verisi için Sliding Window (Kayan Pencere) uygulayan PyTorch Dataset sınıfı."""
    
    def __init__(self, X, y, window_size: int):
        self.X = np.array(X) if not isinstance(X, np.ndarray) else X
        self.y = np.array(y) if not isinstance(y, np.ndarray) else y
        self.window_size = window_size

    def __len__(self):
        return max(0, len(self.X) - self.window_size + 1)

    def __getitem__(self, idx):
        x_window = self.X[idx : idx + self.window_size]
        # Hedef etiket, pencerenin en son zaman adımına karşılık gelen değerdir
        y_label = self.y[idx + self.window_size - 1]

        x_tensor = torch.tensor(x_window, dtype=torch.float32)
        y_tensor = torch.tensor(y_label, dtype=torch.float32)

        return x_tensor, y_tensor


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, window_size: int, batch_size: int):
    """Eğitim, doğrulama ve test verileri için PyTorch DataLoader nesnelerini üretir."""
    
    train_dataset = TimeSeriesDataset(X_train, y_train, window_size)
    val_dataset = TimeSeriesDataset(X_val, y_val, window_size)
    test_dataset = TimeSeriesDataset(X_test, y_test, window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
