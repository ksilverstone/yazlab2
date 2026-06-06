import os
import yaml
from src.pipeline import TimeSeriesPipeline
from src.utils.metrics import save_explainability_log

def main():
    config_path = "configs/config.yaml"
    
    # config.yaml yoksa kodun patlamaması için varsayılan dict oluştur ve kaydet
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        default_config = {
            "experiment": {"log_dir": "logs", "random_seeds": [42, 123, 2026, 7, 999]},
            "data": {
                "skab": {"raw_dir": "data/raw/SKAB", "group_col": "source_file"},
                "batadal": {
                    "file_path": "data/raw/BATADAL/Training_Dataset_2.csv",
                    "train_ratio": 0.6,
                    "val_ratio": 0.2,
                    "test_ratio": 0.2
                }
            },
            "preprocessing": {"scaler": "StandardScaler", "pca_n_components": 1},
            "deep_learning": {
                "max_epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001,
                "early_stopping_patience": 5,
                "models": {
                    "lstm": {"hidden_size": 64, "num_layers": 2, "dropout": 0.2}
                }
            },
            "automata": {
                "paa_size": 4,
                "sax_alphabet_size": 3,
                "window_size": 4,
                "anomaly_threshold": 0.05
            }
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, allow_unicode=True)

    try:
        # Pipeline Kurulumu (Init)
        pipeline = TimeSeriesPipeline(config_path)
        
        # 1. Eğitim (Train): Veri hazırlama, DL eğitimi, SAX/TPM öğrenimi
        pipeline.train_pipeline()
        
        # 2. Test ve Karar (Evaluate): Olasılıksal logların JSON formatında üretimi
        results = pipeline.evaluate_pipeline()
        
        # 3. Klasör Kontrolü ve Dosyaya Yazma
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "explainability_log.json")
        save_explainability_log(results, output_path)
        
        print("İşlem başarıyla tamamlandı, loglar outputs klasörüne kaydedildi.")
        
    except Exception as e:
        # Hata anında konsolu çok kirletmemek adına kısa bir özet (PDF kurallarına istinaden)
        print(f"Bilinmeyen bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()
