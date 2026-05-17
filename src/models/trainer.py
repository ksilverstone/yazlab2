import torch
import copy
from pathlib import Path

class ModelTrainer:
    """Derin öğrenme modelleri için eğitim ve doğrulama döngüsünü (Early Stopping ile) yöneten sınıf."""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, config: dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        
        # Hiperparametreleri config dosyasından okuma
        dl_config = config.get('deep_learning', {})
        self.max_epochs = dl_config.get('max_epochs', 50)
        self.patience = dl_config.get('early_stopping_patience', 5)
        
        # Model kayıt dizini
        self.log_dir = Path(config.get('experiment', {}).get('log_dir', 'logs'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.log_dir / 'best_model.pth'
        
        # Cihaz ayarı (CPU / GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self):
        """Eğitim döngüsünü çalıştırır, early stopping uygular ve en iyi modeli döner."""
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.max_epochs):
            # --- Eğitim (Train) Aşaması ---
            self.model.train()
            
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
            # --- Doğrulama (Validation) Aşaması ---
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    
                    # Tüm batch'in loss toplamını bul
                    val_loss += loss.item() * X_batch.size(0)
                    
            # Ortalama Validation Loss hesabı
            val_loss /= len(self.val_loader.dataset)
            
            # --- Early Stopping ve En İyi Modeli Kaydetme ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Ağırlıkları hem hafızaya (deepcopy) hem de diske alıyoruz
                best_model_state = copy.deepcopy(self.model.state_dict())
                try:
                    torch.save(best_model_state, self.best_model_path)
                except Exception:
                    pass
            else:
                patience_counter += 1
                
            # Belirlenen epoch boyunca iyileşme olmadıysa döngüyü bitir
            if patience_counter >= self.patience:
                break
                
        # Döngü bitiminde elde edilen en iyi ağırlıkları modele geri yükle
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        return self.model
