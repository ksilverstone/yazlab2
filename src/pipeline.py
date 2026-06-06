import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.preprocessor import SKABPreprocessor, BATADALPreprocessor, DataTransformer
from src.data.dataset import create_dataloaders
from src.models.dl_models import build_dl_model
from src.models.trainer import ModelTrainer
from src.models.automata import TimeSeriesSymbolizer, AutomataModel
from src.utils.metrics import compute_metrics, aggregate_seed_results, save_explainability_log


def set_seed(seed: int):
    """Tekrarlanabilirlik için tüm random seed'leri ayarlar."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TimeSeriesPipeline:
    """Zaman Serilerinde Anomali Tespiti projesini uçtan uca yöneten orkestratör sınıf."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.seeds = self.config.get("experiment", {}).get("random_seeds", [42, 123, 2026, 7, 999])
        self.output_dir = self.config.get("experiment", {}).get("output_dir", "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.config.get("experiment", {}).get("log_dir", "logs"), exist_ok=True)

    # ==================== BATADAL Pipeline ====================

    def run_batadal_pipeline(self) -> dict:
        """BATADAL veri seti üzerinde tüm modelleri tüm seed'lerle çalıştırır."""
        preprocessor = BATADALPreprocessor(self.config_path)
        X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.load_and_split_data()

        if X_train.empty:
            return {}

        all_results = {}

        # --- DL Modelleri ---
        for model_name in ["lstm", "gru", "cnn_1d"]:
            seed_metrics = []
            for seed_idx, seed in enumerate(self.seeds):
                print(f"BATADAL - {model_name} eğitiliyor (Seed {seed_idx+1}/{len(self.seeds)})...")
                set_seed(seed)
                transformer = DataTransformer(self.config_path)

                X_train_dl, _ = transformer.fit_transform(X_train)
                X_val_dl, _ = transformer.transform(X_val)
                X_test_dl, _ = transformer.transform(X_test)

                window_size = self.config.get("automata", {}).get("window_size", 4)
                batch_size = self.config.get("deep_learning", {}).get("batch_size", 32)

                train_loader, val_loader, test_loader = create_dataloaders(
                    X_train_dl,
                    y_train,
                    X_val_dl,
                    y_val,
                    X_test_dl,
                    y_test,
                    window_size=window_size,
                    batch_size=batch_size,
                )

                input_size = X_train_dl.shape[1]
                model = build_dl_model(model_name, input_size, self.config)

                optimizer = optim.Adam(
                    model.parameters(),
                    lr=self.config.get("deep_learning", {}).get("learning_rate", 0.001),
                )
                criterion = nn.CrossEntropyLoss()

                trainer = ModelTrainer(
                    model, train_loader, val_loader, criterion, optimizer, self.config
                )
                trained_model = trainer.train()

                # Test üzerinde tahmin
                y_pred, y_true_aligned = self._predict_dl(trained_model, test_loader)
                metrics = compute_metrics(y_true_aligned, y_pred)
                seed_metrics.append(metrics)

            all_results[model_name] = aggregate_seed_results(seed_metrics)
            all_results[model_name]["raw_seed_f1"] = [m.get("f1_score", 0) for m in seed_metrics]

        # --- Automata Modeli ---
        automata_seed_metrics = []
        automata_explanations = []
        for seed_idx, seed in enumerate(self.seeds):
            print(f"BATADAL - automata eğitiliyor (Seed {seed_idx+1}/{len(self.seeds)})...")
            set_seed(seed)
            transformer = DataTransformer(self.config_path)
            _, X_train_auto = transformer.fit_transform(X_train)
            _, X_test_auto = transformer.transform(X_test)

            paa_size = self.config.get("automata", {}).get("paa_size", 4)
            alphabet_size = self.config.get("automata", {}).get("sax_alphabet_size", 3)
            window_size = self.config.get("automata", {}).get("window_size", 4)
            anomaly_threshold = self.config.get("automata", {}).get("anomaly_threshold", 0.05)

            symbolizer = TimeSeriesSymbolizer(paa_size=paa_size, alphabet_size=alphabet_size)
            automata = AutomataModel(window_size=window_size)

            X_train_1d = X_train_auto.flatten()
            symbolizer.fit(X_train_1d)
            sax_train = symbolizer.transform(X_train_1d)
            automata.fit(sax_train)

            X_test_1d = X_test_auto.flatten()
            sax_test = symbolizer.transform(X_test_1d)

            predictions = automata.predict_labels(sax_test, anomaly_threshold)
            explanations = automata.predict_and_explain(sax_test, anomaly_threshold)

            # Automata çıktısı DL window'lu çıktıdan farklı uzunlukta olabileceğinden
            # y_test'i uygun şekilde kırpıyoruz
            n_preds = len(predictions)
            y_test_arr = np.array(y_test)
            if n_preds < len(y_test_arr):
                y_aligned = y_test_arr[:n_preds]
            else:
                y_aligned = y_test_arr

            metrics = compute_metrics(y_aligned, predictions[: len(y_aligned)])
            automata_seed_metrics.append(metrics)
            automata_explanations = explanations

        all_results["automata"] = aggregate_seed_results(automata_seed_metrics)
        all_results["automata"]["raw_seed_f1"] = [
            m.get("f1_score", 0) for m in automata_seed_metrics
        ]

        # Açıklanabilirlik loglarını kaydet
        save_explainability_log(
            automata_explanations, os.path.join(self.output_dir, "batadal_explainability.json")
        )

        return all_results

    # ==================== SKAB Pipeline ====================

    def run_skab_pipeline(self) -> dict:
        """SKAB veri seti üzerinde GroupKFold ile tüm modelleri çalıştırır."""
        preprocessor = SKABPreprocessor(self.config_path)
        X, y, groups = preprocessor.load_and_merge_data()

        if X.empty:
            return {}

        splits = preprocessor.get_group_kfold_splits(X, y, groups, n_splits=5)
        all_results = {}

        # --- DL Modelleri ---
        for model_name in ["lstm", "gru", "cnn_1d"]:
            seed_metrics = []
            for seed in self.seeds:
                fold_metrics = []
                for fold_idx, (train_idx, test_idx) in enumerate(splits):
                    print(f"SKAB - {model_name} eğitiliyor (Seed {seed}, Fold {fold_idx+1}/{len(splits)})...")
                    set_seed(seed)

                    X_train_fold = X.iloc[train_idx]
                    y_train_fold = y.iloc[train_idx]
                    X_test_fold = X.iloc[test_idx]
                    y_test_fold = y.iloc[test_idx]

                    # Train'i %80 train %20 val olarak böl
                    n_train = len(X_train_fold)
                    val_split = int(n_train * 0.8)
                    X_tr = X_train_fold.iloc[:val_split]
                    y_tr = y_train_fold.iloc[:val_split]
                    X_vl = X_train_fold.iloc[val_split:]
                    y_vl = y_train_fold.iloc[val_split:]

                    transformer = DataTransformer(self.config_path)
                    X_tr_dl, _ = transformer.fit_transform(X_tr)
                    X_vl_dl, _ = transformer.transform(X_vl)
                    X_te_dl, _ = transformer.transform(X_test_fold)

                    window_size = self.config.get("automata", {}).get("window_size", 4)
                    batch_size = self.config.get("deep_learning", {}).get("batch_size", 32)

                    train_loader, val_loader, test_loader = create_dataloaders(
                        X_tr_dl,
                        y_tr,
                        X_vl_dl,
                        y_vl,
                        X_te_dl,
                        y_test_fold,
                        window_size=window_size,
                        batch_size=batch_size,
                    )

                    input_size = X_tr_dl.shape[1]
                    model = build_dl_model(model_name, input_size, self.config)

                    optimizer = optim.Adam(
                        model.parameters(),
                        lr=self.config.get("deep_learning", {}).get("learning_rate", 0.001),
                    )
                    criterion = nn.CrossEntropyLoss()
                    trainer = ModelTrainer(
                        model, train_loader, val_loader, criterion, optimizer, self.config
                    )
                    trained_model = trainer.train()

                    y_pred, y_true_aligned = self._predict_dl(trained_model, test_loader)
                    metrics = compute_metrics(y_true_aligned, y_pred)
                    fold_metrics.append(metrics)

                # Fold ortalaması
                avg_fold = aggregate_seed_results(fold_metrics)
                seed_metrics.append({k: v["mean"] for k, v in avg_fold.items()})

            all_results[model_name] = aggregate_seed_results(seed_metrics)
            all_results[model_name]["raw_seed_f1"] = [m.get("f1_score", 0) for m in seed_metrics]

        # --- Automata Modeli ---
        automata_seed_metrics = []
        for seed in self.seeds:
            fold_metrics = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                print(f"SKAB - automata eğitiliyor (Seed {seed}, Fold {fold_idx+1}/{len(splits)})...")
                set_seed(seed)

                X_train_fold = X.iloc[train_idx]
                X_test_fold = X.iloc[test_idx]
                y_test_fold = y.iloc[test_idx]

                transformer = DataTransformer(self.config_path)
                _, X_train_auto = transformer.fit_transform(X_train_fold)
                _, X_test_auto = transformer.transform(X_test_fold)

                paa_size = self.config.get("automata", {}).get("paa_size", 4)
                alphabet_size = self.config.get("automata", {}).get("sax_alphabet_size", 3)
                window_size = self.config.get("automata", {}).get("window_size", 4)
                anomaly_threshold = self.config.get("automata", {}).get("anomaly_threshold", 0.05)

                symbolizer = TimeSeriesSymbolizer(paa_size=paa_size, alphabet_size=alphabet_size)
                automata = AutomataModel(window_size=window_size)

                X_train_1d = X_train_auto.flatten()
                symbolizer.fit(X_train_1d)
                sax_train = symbolizer.transform(X_train_1d)
                automata.fit(sax_train)

                X_test_1d = X_test_auto.flatten()
                sax_test = symbolizer.transform(X_test_1d)

                predictions = automata.predict_labels(sax_test, anomaly_threshold)

                n_preds = len(predictions)
                y_test_arr = np.array(y_test_fold)
                if n_preds < len(y_test_arr):
                    y_aligned = y_test_arr[:n_preds]
                else:
                    y_aligned = y_test_arr

                metrics = compute_metrics(y_aligned, predictions[: len(y_aligned)])
                fold_metrics.append(metrics)

            avg_fold = aggregate_seed_results(fold_metrics)
            automata_seed_metrics.append({k: v["mean"] for k, v in avg_fold.items()})

        all_results["automata"] = aggregate_seed_results(automata_seed_metrics)
        all_results["automata"]["raw_seed_f1"] = [
            m.get("f1_score", 0) for m in automata_seed_metrics
        ]
        return all_results

    # ==================== Gürültü Deneyi ====================

    def run_noise_experiment(self, dataset_name: str = "batadal") -> dict:
        """Gaussian gürültü eklenmiş veri üzerinde model performansını test eder."""
        noise_std = self.config.get("automata", {}).get("noise_std", 0.1)

        if dataset_name == "batadal":
            preprocessor = BATADALPreprocessor(self.config_path)
            X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.load_and_split_data()
            if X_train.empty:
                return {}

            # Test verisine Gaussian gürültü ekle
            np.random.seed(42)
            X_test_noisy = X_test.copy()
            noise = np.random.normal(0, noise_std, X_test_noisy.shape)
            X_test_noisy = X_test_noisy + noise

            results = {}
            for model_name in ["lstm", "gru", "cnn_1d"]:
                seed = self.seeds[0]
                set_seed(seed)

                transformer = DataTransformer(self.config_path)
                X_train_dl, _ = transformer.fit_transform(X_train)
                X_val_dl, _ = transformer.transform(X_val)
                X_test_dl, _ = transformer.transform(X_test_noisy)

                window_size = self.config.get("automata", {}).get("window_size", 4)
                batch_size = self.config.get("deep_learning", {}).get("batch_size", 32)

                train_loader, val_loader, test_loader = create_dataloaders(
                    X_train_dl,
                    y_train,
                    X_val_dl,
                    y_val,
                    X_test_dl,
                    y_test,
                    window_size=window_size,
                    batch_size=batch_size,
                )

                input_size = X_train_dl.shape[1]
                model = build_dl_model(model_name, input_size, self.config)
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=self.config.get("deep_learning", {}).get("learning_rate", 0.001),
                )
                criterion = nn.CrossEntropyLoss()
                trainer = ModelTrainer(
                    model, train_loader, val_loader, criterion, optimizer, self.config
                )
                trained_model = trainer.train()

                y_pred, y_true_aligned = self._predict_dl(trained_model, test_loader)
                results[model_name] = compute_metrics(y_true_aligned, y_pred)

            # Automata noise
            set_seed(42)
            transformer = DataTransformer(self.config_path)
            _, X_train_auto = transformer.fit_transform(X_train)
            _, X_test_auto = transformer.transform(X_test_noisy)

            paa_size = self.config.get("automata", {}).get("paa_size", 4)
            alphabet_size = self.config.get("automata", {}).get("sax_alphabet_size", 3)
            window_size = self.config.get("automata", {}).get("window_size", 4)
            anomaly_threshold = self.config.get("automata", {}).get("anomaly_threshold", 0.05)

            symbolizer = TimeSeriesSymbolizer(paa_size=paa_size, alphabet_size=alphabet_size)
            automata_model = AutomataModel(window_size=window_size)
            symbolizer.fit(X_train_auto.flatten())
            sax_train = symbolizer.transform(X_train_auto.flatten())
            automata_model.fit(sax_train)

            sax_test = symbolizer.transform(X_test_auto.flatten())
            preds = automata_model.predict_labels(sax_test, anomaly_threshold)
            y_test_arr = np.array(y_test)
            preds = preds[: len(y_test_arr)]
            y_aligned = y_test_arr[: len(preds)]
            results["automata"] = compute_metrics(y_aligned, preds)

            return results

        return {}

    # ==================== Parametre Varyasyonu ====================

    def run_parameter_variation(self, dataset_name: str = "batadal") -> dict:
        """window_size ve alphabet_size varyasyonlarının automata performansına etkisini analiz eder."""
        variations = self.config.get("automata", {}).get("parameter_variations", {})
        window_sizes = variations.get("window_sizes", [3, 4, 5, 6])
        alphabet_sizes = variations.get("alphabet_sizes", [3, 4, 5, 6])

        if dataset_name == "batadal":
            preprocessor = BATADALPreprocessor(self.config_path)
            X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.load_and_split_data()
            if X_train.empty:
                return {}

            set_seed(42)
            transformer = DataTransformer(self.config_path)
            _, X_train_auto = transformer.fit_transform(X_train)
            _, X_test_auto = transformer.transform(X_test)

            results = {"window_size": {}, "alphabet_size": {}}
            anomaly_threshold = self.config.get("automata", {}).get("anomaly_threshold", 0.05)

            # Window size varyasyonu (alphabet sabit = 3)
            for ws in window_sizes:
                paa_size = self.config.get("automata", {}).get("paa_size", 4)
                symbolizer = TimeSeriesSymbolizer(paa_size=paa_size, alphabet_size=3)
                automata = AutomataModel(window_size=ws)

                symbolizer.fit(X_train_auto.flatten())
                sax_train = symbolizer.transform(X_train_auto.flatten())
                automata.fit(sax_train)

                sax_test = symbolizer.transform(X_test_auto.flatten())
                preds = automata.predict_labels(sax_test, anomaly_threshold)

                y_test_arr = np.array(y_test)
                preds = preds[: len(y_test_arr)]
                y_aligned = y_test_arr[: len(preds)]

                metrics = compute_metrics(y_aligned, preds)
                metrics["state_count"] = automata.get_state_count()
                metrics["transition_density"] = automata.get_transition_density()
                results["window_size"][ws] = metrics

            # Alphabet size varyasyonu (window sabit = 4)
            for als in alphabet_sizes:
                symbolizer = TimeSeriesSymbolizer(paa_size=4, alphabet_size=als)
                automata = AutomataModel(window_size=4)

                symbolizer.fit(X_train_auto.flatten())
                sax_train = symbolizer.transform(X_train_auto.flatten())
                automata.fit(sax_train)

                sax_test = symbolizer.transform(X_test_auto.flatten())
                preds = automata.predict_labels(sax_test, anomaly_threshold)

                y_test_arr = np.array(y_test)
                preds = preds[: len(y_test_arr)]
                y_aligned = y_test_arr[: len(preds)]

                metrics = compute_metrics(y_aligned, preds)
                metrics["state_count"] = automata.get_state_count()
                metrics["transition_density"] = automata.get_transition_density()
                results["alphabet_size"][als] = metrics

            return results

        return {}

    # ==================== Yardımcı Metotlar ====================

    @staticmethod
    def _predict_dl(model, test_loader) -> tuple:
        """DL modelinin test loader üzerindeki tahminlerini ve gerçek etiketlerini döner."""
        device = next(model.parameters()).device
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        return np.array(all_preds), np.array(all_labels)
