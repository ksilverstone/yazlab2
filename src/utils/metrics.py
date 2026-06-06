import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy, Precision, Recall, F1-score hesaplar ve dict olarak döner."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
    }


def compute_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Confusion matrix hesaplar."""
    return confusion_matrix(y_true, y_pred)


def compute_classification_report(y_true, y_pred) -> str:
    """Detaylı sınıflandırma raporu döner."""
    return classification_report(
        y_true, y_pred, target_names=["Normal", "Anomaly"], zero_division=0
    )


def aggregate_seed_results(all_seed_metrics: list) -> dict:
    """Birden fazla seed sonucunun ortalamasını ve standart sapmasını hesaplar."""
    if not all_seed_metrics:
        return {}

    keys = all_seed_metrics[0].keys()
    aggregated = {}

    for key in keys:
        values = [m[key] for m in all_seed_metrics]
        aggregated[key] = {
            "mean": round(float(np.mean(values)), 6),
            "std": round(float(np.std(values)), 6),
        }

    return aggregated


def save_explainability_log(results: list, output_path: str):
    """Pipeline'dan dönen açıklanabilirlik log listesini JSON formatında kaydeder."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def save_experiment_results(results: dict, output_path: str):
    """Deney sonuçlarını JSON formatında kaydeder."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4, default=str)
