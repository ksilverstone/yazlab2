import yaml
import torch.nn as nn
import torch.optim as optim

from src.data.preprocessor import BATADALPreprocessor, DataTransformer
from src.data.dataset import create_dataloaders
from src.models.dl_models import AnomalyLSTM
from src.models.trainer import ModelTrainer
from src.models.automata import TimeSeriesSymbolizer, AutomataModel

class TimeSeriesPipeline:
    """Zaman Serilerinde Anomali Tespiti ve Açıklanabilir AI projesini uçtan uca yöneten orkestratör sınıf."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # 1. Ön İşleme ve Dönüşüm Sınıflarını Başlat (Initialize)
        self.preprocessor = BATADALPreprocessor(self.config_path)
        self.transformer = DataTransformer(self.config_path)
        
        # 2. Automata Parametreleri ve Nesneleri
        paa_size = self.config.get('automata', {}).get('paa_size', 4)
        alphabet_size = self.config.get('automata', {}).get('sax_alphabet_size', 3)
        self.window_size = self.config.get('automata', {}).get('window_size', 4)
        
        self.symbolizer = TimeSeriesSymbolizer(paa_size=paa_size, alphabet_size=alphabet_size)
        self.automata = AutomataModel(window_size=self.window_size)
        
        # İleride kullanılacak test değişkenleri
        self.X_test = None
        self.y_test = None
        
    def train_pipeline(self):
        """Veriyi okur, Derin Öğrenme modellerini ve Olasılıksal Automata yapısını sırayla eğitir."""
        
        # 1. Veri Okuma, Temizleme ve Kronolojik Bölme
        X_train, y_train, X_val, y_val, X_test, y_test = self.preprocessor.load_and_split_data()
        
        # Test işlemlerine passlamak için bellekte tut
        self.X_test = X_test
        self.y_test = y_test
        
        # 2. Normalize Etme ve PCA (Sızıntı Korumalı)
        X_train_dl, X_train_auto = self.transformer.fit_transform(X_train)
        X_val_dl, X_val_auto = self.transformer.transform(X_val)
        X_test_dl, _ = self.transformer.transform(X_test)
        
        # 3. PyTorch Dataloader'ları Üretme
        batch_size = self.config.get('deep_learning', {}).get('batch_size', 32)
        train_loader, val_loader, _ = create_dataloaders(
            X_train_dl, y_train, X_val_dl, y_val, X_test_dl, y_test,
            window_size=self.window_size, batch_size=batch_size
        )
        
        # 4. Derin Öğrenme Modeli (Örn: LSTM) Eğitim Adımları
        input_size = X_train_dl.shape[1]
        dl_config = self.config.get('deep_learning', {}).get('models', {}).get('lstm', {})
        
        dl_model = AnomalyLSTM(
            input_size=input_size, 
            hidden_size=dl_config.get('hidden_size', 64), 
            num_layers=dl_config.get('num_layers', 2),
            dropout=dl_config.get('dropout', 0.2)
        )
        
        optimizer = optim.Adam(dl_model.parameters(), lr=self.config.get('deep_learning', {}).get('learning_rate', 0.001))
        criterion = nn.CrossEntropyLoss()
        
        trainer = ModelTrainer(dl_model, train_loader, val_loader, criterion, optimizer, self.config)
        trainer.train() # Model arka planda sessizce eğitilir ve kaydedilir
        
        # 5. Automata Modelinin Eğitilmesi (SAX, PAA ve TPM)
        # PCA'dan çıkan 1D veriyi tek eksene (flatten) indirge
        X_train_1d = X_train_auto.flatten()
        
        # Sınırları belirle (Fit)
        self.symbolizer.fit(X_train_1d)
        
        # Eğitim setini devasa bir string'e dönüştür (Transform)
        sax_string_train = self.symbolizer.transform(X_train_1d)
        
        # SAX stringi üzerinden pencereleri dolaşarak geçiş olasılıklarını (Markov/TPM) hesapla
        self.automata.fit(sax_string_train)

    def evaluate_pipeline(self) -> list:
        """Eğitilmiş yapıyı test verisi üzerinden koşturarak açıklanabilirlik loglarını listeler."""
        if self.X_test is None:
            return []
            
        # 1. Test verisini sadece transforme et ve 1D yapısına büründür
        _, X_test_auto = self.transformer.transform(self.X_test)
        X_test_1d = X_test_auto.flatten()
        
        # 2. Test verisini SAX (harf) formatına çevir
        sax_string_test = self.symbolizer.transform(X_test_1d)
        
        # 3. Model üzerinden karar verme işlemlerini koştur ve JSON loglarını al
        anomaly_threshold = self.config.get('automata', {}).get('anomaly_threshold', 0.05)
        explainability_logs = self.automata.predict_and_explain(sax_string_test, anomaly_threshold=anomaly_threshold)
        
        return explainability_logs
