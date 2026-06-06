"""
Deney sonuçları tamamlandıktan sonra çalıştırılacak rapor üretme scripti.
Görselleştirmeler, istatistiksel testler ve özet tabloları üretir.
"""

import os
import json
import yaml
import numpy as np

from src.pipeline import TimeSeriesPipeline, set_seed
from src.data.preprocessor import BATADALPreprocessor, SKABPreprocessor, DataTransformer
from src.data.dataset import create_dataloaders
from src.models.dl_models import build_dl_model
from src.models.automata import TimeSeriesSymbolizer, AutomataModel
from src.models.trainer import ModelTrainer
from src.utils.metrics import compute_metrics, compute_confusion_matrix, save_experiment_results
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_transition_heatmap,
    plot_parameter_sensitivity,
    plot_model_comparison,
    plot_precision_recall_curve,
)

import torch
import torch.nn as nn
import torch.optim as optim


def generate_confusion_matrices(config_path: str, output_dir: str):
    """Her model için confusion matrix üretir ve kaydeder."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    preprocessor = BATADALPreprocessor(config_path)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.load_and_split_data()
    if X_train.empty:
        return

    cm_dir = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    # DL modelleri
    for model_name in ["lstm", "gru", "cnn_1d"]:
        set_seed(42)
        transformer = DataTransformer(config_path)
        X_train_dl, _ = transformer.fit_transform(X_train)
        X_val_dl, _ = transformer.transform(X_val)
        X_test_dl, _ = transformer.transform(X_test)

        window_size = config.get("automata", {}).get("window_size", 4)
        batch_size = config.get("deep_learning", {}).get("batch_size", 32)

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
        model = build_dl_model(model_name, input_size, config)
        optimizer = optim.Adam(
            model.parameters(), lr=config.get("deep_learning", {}).get("learning_rate", 0.001)
        )
        criterion = nn.CrossEntropyLoss()
        trainer = ModelTrainer(model, train_loader, val_loader, criterion, optimizer, config)
        trained_model = trainer.train()

        y_pred, y_true = TimeSeriesPipeline._predict_dl(trained_model, test_loader)
        cm = compute_confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(
            cm,
            title=f"Confusion Matrix - {model_name.upper()} (BATADAL)",
            save_path=os.path.join(cm_dir, f"cm_{model_name}_batadal.png"),
        )
        print(f"  CM kaydedildi: {model_name}")

    # Automata
    set_seed(42)
    transformer = DataTransformer(config_path)
    _, X_train_auto = transformer.fit_transform(X_train)
    _, X_test_auto = transformer.transform(X_test)

    paa_size = config.get("automata", {}).get("paa_size", 4)
    alphabet_size = config.get("automata", {}).get("sax_alphabet_size", 3)
    window_size = config.get("automata", {}).get("window_size", 4)
    anomaly_threshold = config.get("automata", {}).get("anomaly_threshold", 0.05)

    symbolizer = TimeSeriesSymbolizer(paa_size=paa_size, alphabet_size=alphabet_size)
    automata = AutomataModel(window_size=window_size)
    symbolizer.fit(X_train_auto.flatten())
    sax_train = symbolizer.transform(X_train_auto.flatten())
    automata.fit(sax_train)

    sax_test = symbolizer.transform(X_test_auto.flatten())
    preds = automata.predict_labels(sax_test, anomaly_threshold)

    y_test_arr = np.array(y_test)
    min_len = min(len(preds), len(y_test_arr))
    cm = compute_confusion_matrix(y_test_arr[:min_len], preds[:min_len])
    plot_confusion_matrix(
        cm,
        title="Confusion Matrix - Automata (BATADAL)",
        save_path=os.path.join(cm_dir, "cm_automata_batadal.png"),
    )

    # Transition heatmap ve State Diagram
    plot_transition_heatmap(
        automata.transition_matrix,
        title="Transition Probability Heatmap",
        save_path=os.path.join(output_dir, "transition_heatmap.png"),
    )
    print("  Transition heatmap kaydedildi")

    from src.utils.visualization import plot_automata_state_diagram

    plot_automata_state_diagram(
        automata.transition_matrix,
        title="Automata State Diagram",
        save_path=os.path.join(output_dir, "automata_state_diagram.png"),
    )
    print("  State diagram kaydedildi")


def generate_parameter_plots(config_path: str, output_dir: str):
    """Parametre varyasyonu grafiklerini üretir."""
    results_path = os.path.join(output_dir, "all_experiment_results.json")
    if not os.path.exists(results_path):
        print("  Deney sonuçları bulunamadı, parametre grafikleri atlanıyor.")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    param_results = all_results.get("batadal_param_variation", {})
    plots_dir = os.path.join(output_dir, "parameter_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Window size
    ws_data = param_results.get("window_size", {})
    if ws_data:
        plot_parameter_sensitivity(
            ws_data,
            "Window Size",
            "f1_score",
            title="Window Size vs F1-Score",
            save_path=os.path.join(plots_dir, "window_size_sensitivity.png"),
        )

    # Alphabet size
    as_data = param_results.get("alphabet_size", {})
    if as_data:
        plot_parameter_sensitivity(
            as_data,
            "Alphabet Size",
            "f1_score",
            title="Alphabet Size vs F1-Score",
            save_path=os.path.join(plots_dir, "alphabet_size_sensitivity.png"),
        )

    print("  Parametre grafikleri kaydedildi")


def generate_model_comparison(config_path: str, output_dir: str):
    """Model karşılaştırma grafiklerini üretir."""
    results_path = os.path.join(output_dir, "all_experiment_results.json")
    if not os.path.exists(results_path):
        return

    with open(results_path) as f:
        all_results = json.load(f)

    plots_dir = os.path.join(output_dir, "comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)

    for dataset_key in ["batadal_base", "skab_base"]:
        dataset_results = all_results.get(dataset_key, {})
        if not dataset_results:
            continue

        comparison = {}
        for model_name, metrics in dataset_results.items():
            if isinstance(metrics, dict) and "f1_score" in metrics:
                f1_data = metrics["f1_score"]
                if isinstance(f1_data, dict):
                    comparison[model_name] = f1_data.get("mean", 0)
                else:
                    comparison[model_name] = f1_data

        if comparison:
            dataset_label = dataset_key.replace("_base", "").upper()
            plot_model_comparison(
                {k: {"f1_score": v} for k, v in comparison.items()},
                metric_name="f1_score",
                title=f"Model Karşılaştırması - {dataset_label} (F1-Score)",
                save_path=os.path.join(plots_dir, f"model_comparison_{dataset_key}.png"),
            )

    print("  Model karşılaştırma grafikleri kaydedildi")


def run_statistical_tests(config_path: str, output_dir: str):
    """Wilcoxon signed-rank test uygular."""
    from scipy.stats import wilcoxon

    results_path = os.path.join(output_dir, "all_experiment_results.json")
    if not os.path.exists(results_path):
        return {}

    with open(results_path) as f:
        all_results = json.load(f)

    stat_results = {}

    for dataset_key in ["batadal_base", "skab_base"]:
        dataset_data = all_results.get(dataset_key, {})
        if not dataset_data:
            continue

        stat_results[dataset_key] = {}
        models = list(dataset_data.keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]

                # Raw listeleri al
                raw_m1 = dataset_data[m1].get("raw_seed_f1", [])
                raw_m2 = dataset_data[m2].get("raw_seed_f1", [])

                f1_mean_m1 = (
                    dataset_data[m1].get("f1_score", {}).get("mean", 0)
                    if isinstance(dataset_data[m1].get("f1_score"), dict)
                    else dataset_data[m1].get("f1_score", 0)
                )
                f1_mean_m2 = (
                    dataset_data[m2].get("f1_score", {}).get("mean", 0)
                    if isinstance(dataset_data[m2].get("f1_score"), dict)
                    else dataset_data[m2].get("f1_score", 0)
                )

                res = {
                    f"{m1}_f1_mean": f1_mean_m1,
                    f"{m2}_f1_mean": f1_mean_m2,
                    "difference": abs(f1_mean_m1 - f1_mean_m2),
                }

                # Wilcoxon için iki listenin en az 2 elemanlı olması gerekir
                if len(raw_m1) > 1 and len(raw_m2) > 1 and len(raw_m1) == len(raw_m2):
                    if raw_m1 == raw_m2:
                        res["wilcoxon_p_value"] = 1.0
                        res["statistically_significant"] = False
                    else:
                        try:
                            w_stat, p_val = wilcoxon(raw_m1, raw_m2, zero_method="zsplit")
                            res["wilcoxon_p_value"] = p_val
                            res["statistically_significant"] = bool(p_val < 0.05)
                        except Exception as e:
                            res["wilcoxon_p_value"] = None
                            res["statistically_significant"] = False
                            res["error"] = str(e)
                else:
                    res["wilcoxon_p_value"] = None
                    res["statistically_significant"] = False

                stat_results[dataset_key][f"{m1}_vs_{m2}"] = res

    save_path = os.path.join(output_dir, "statistical_tests.json")
    with open(save_path, "w") as f:
        json.dump(stat_results, f, indent=4)

    print("  İstatistiksel test sonuçları kaydedildi")
    return stat_results


def main():
    config_path = "configs/config.yaml"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Rapor ve Görselleştirme Üretimi Başlıyor...")
    print("=" * 60)

    print("\n1. Confusion Matrix'ler üretiliyor...")
    generate_confusion_matrices(config_path, output_dir)

    print("\n2. Parametre grafikleri üretiliyor...")
    generate_parameter_plots(config_path, output_dir)

    print("\n3. Model karşılaştırma grafikleri üretiliyor...")
    generate_model_comparison(config_path, output_dir)

    print("\n4. İstatistiksel testler çalıştırılıyor...")
    run_statistical_tests(config_path, output_dir)

    print("\n" + "=" * 60)
    print(f"Tüm raporlar '{output_dir}/' altına kaydedildi.")
    print("=" * 60)


if __name__ == "__main__":
    main()
