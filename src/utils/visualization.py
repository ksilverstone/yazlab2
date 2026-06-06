import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None):
    """Confusion matrix'i matplotlib ile çizer ve kaydeder."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        ax=ax,
    )
    ax.set_xlabel("Tahmin Edilen")
    ax.set_ylabel("Gerçek")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_scores, title="ROC Curve", save_path=None):
    """ROC eğrisini çizer."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_precision_recall_curve(y_true, y_scores, title="Precision-Recall Curve", save_path=None):
    """Precision-Recall eğrisini çizer."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="green", lw=2, label=f"AP = {avg_precision:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_transition_heatmap(
    transition_matrix: dict, title="Transition Probability Heatmap", save_path=None
):
    """Geçiş olasılık matrisini heatmap olarak çizer."""
    states = sorted(
        set(
            list(transition_matrix.keys())
            + [s for v in transition_matrix.values() for s in v.keys()]
        )
    )

    if len(states) > 30:
        states = sorted(transition_matrix.keys())[:30]

    n = len(states)
    matrix = np.zeros((n, n))
    state_idx = {s: i for i, s in enumerate(states)}

    for src, targets in transition_matrix.items():
        if src not in state_idx:
            continue
        for tgt, prob in targets.items():
            if tgt in state_idx:
                matrix[state_idx[src]][state_idx[tgt]] = prob

    fig, ax = plt.subplots(figsize=(max(8, n * 0.5), max(6, n * 0.4)))
    sns.heatmap(
        matrix,
        xticklabels=states,
        yticklabels=states,
        cmap="YlOrRd",
        annot=(n <= 15),
        fmt=".2f",
        ax=ax,
    )
    ax.set_xlabel("Hedef State")
    ax.set_ylabel("Kaynak State")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_parameter_sensitivity(
    results_dict: dict, param_name: str, metric_name: str = "f1_score", title=None, save_path=None
):
    """Parametre duyarlılık grafiği çizer. F1-score ve State Count içerir."""
    if title is None:
        title = f"{param_name} Sensitivity"

    param_values = sorted(results_dict.keys())
    x_labels = [str(p) for p in param_values]
    metric_values = [results_dict[p].get(metric_name, 0) for p in param_values]
    state_counts = [results_dict[p].get("state_count", 0) for p in param_values]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:blue"
    ax1.set_xlabel(param_name)
    ax1.set_ylabel(metric_name, color=color)
    ax1.plot(x_labels, metric_values, marker="o", linewidth=2, markersize=8, color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("State Count", color=color)
    ax2.bar(x_labels, state_counts, alpha=0.3, color=color, width=0.4)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(title)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(
    results: dict, metric_name: str = "f1_score", title="Model Karşılaştırması", save_path=None
):
    """Tüm modellerin bir metriğini bar chart olarak karşılaştırır."""
    model_names = list(results.keys())
    values = [results[m].get(metric_name, 0) for m in model_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(model_names, values, color=["#2196F3", "#4CAF50", "#FF9800", "#E91E63"])
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_automata_state_diagram(transition_matrix: dict, title="Automata State Diagram", save_path=None):
    """NetworkX ile State Geçiş Grafiği Çizer."""
    
    G = nx.DiGraph()
    for src, targets in transition_matrix.items():
        for tgt, prob in targets.items():
            if prob > 0.05:
                G.add_edge(src, tgt, weight=prob)
                
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', 
            node_size=1500, font_size=9, font_weight='bold', 
            edge_color='gray', arrows=True)
            
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    ax.set_title(title)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
