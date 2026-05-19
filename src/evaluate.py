"""
src/evaluate.py
---------------
Visualisation functions for model evaluation.
All plots saved to data/plots/ for use in reports and dashboard.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc

os.makedirs("data/plots", exist_ok=True)

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False


# ──────────────────────────────────────────────────
# 1. CONFUSION MATRIX
# ──────────────────────────────────────────────────

def plot_confusion_matrix(metrics: dict, save: bool = True) -> None:
    """
    Plot confusion matrix with annotations explaining each cell.

        Predicted No Default | Predicted Default
    ────────────────────────┼──────────────────────
    Actual No Default   TN  |        FP
    Actual Default      FN  |        TP

    TN = True Negative  → correctly approved good borrower
    FP = False Positive → wrongly flagged good borrower
    FN = False Negative → MISSED a defaulter (most costly!)
    TP = True Positive  → correctly flagged defaulter
    """
    cm   = np.array(metrics["conf_matrix"])
    name = metrics["model_name"]

    labels = np.array([
        [f"TN\n{cm[0,0]}\nCorrect approval",   f"FP\n{cm[0,1]}\nWrong rejection"],
        [f"FN\n{cm[1,0]}\nMissed default!",     f"TP\n{cm[1,1]}\nCaught default"],
    ])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues",
        xticklabels=["Pred: No Default", "Pred: Default"],
        yticklabels=["True: No Default", "True: Default"],
        ax=ax, linewidths=1, linecolor="white",
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax.set_title(f"{name} — Confusion Matrix", fontweight="bold", pad=14)
    ax.set_ylabel("Actual Label", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")

    plt.tight_layout()
    if save:
        path = f"data/plots/confusion_matrix_{name.lower().replace(' ', '_')}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


# ──────────────────────────────────────────────────
# 2. ROC CURVE — COMPARE BOTH MODELS
# ──────────────────────────────────────────────────

def plot_roc_curves(metrics_list: list, save: bool = True) -> None:
    """
    Plot ROC curves for multiple models on the same axis.

    ROC Curve interpretation:
    - X axis: False Positive Rate (FPR) = FP / (FP + TN)
      → how often we wrongly reject a good borrower
    - Y axis: True Positive Rate (TPR) = TP / (TP + FN)
      → how often we catch a real defaulter
    - Diagonal line = random guessing (AUC = 0.5)
    - Perfect model = top-left corner (AUC = 1.0)
    - We want to be as far into the top-left as possible
    """
    colors = ["#534AB7", "#D85A30", "#1D9E75", "#D4537E"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, metrics in enumerate(metrics_list):
        fpr = metrics["roc_curve"]["fpr"]
        tpr = metrics["roc_curve"]["tpr"]
        roc_auc = metrics["roc_auc"]
        name = metrics["model_name"]

        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                linewidth=2.5, label=f"{name}  (AUC = {roc_auc:.4f})")

    # Random baseline
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.50)")

    ax.fill_between(
        metrics_list[-1]["roc_curve"]["fpr"],
        metrics_list[-1]["roc_curve"]["tpr"],
        alpha=0.06, color=colors[len(metrics_list) - 1]
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate  (FPR)", fontweight="bold")
    ax.set_ylabel("True Positive Rate  (TPR)", fontweight="bold")
    ax.set_title("ROC Curve Comparison", fontweight="bold", pad=14)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if save:
        plt.savefig("data/plots/roc_curves.png", dpi=150, bbox_inches="tight")
        print("Saved: data/plots/roc_curves.png")
    plt.show()


# ──────────────────────────────────────────────────
# 3. METRICS COMPARISON BAR CHART
# ──────────────────────────────────────────────────

def plot_metrics_comparison(metrics_list: list, save: bool = True) -> None:
    """
    Side-by-side bar chart comparing all key metrics
    across both models.
    """
    metric_keys   = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    colors        = ["#534AB7", "#D85A30"]

    x     = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, metrics in enumerate(metrics_list):
        values = [metrics[k] for k in metric_keys]
        bars = ax.bar(
            x + i * width, values, width,
            label=metrics["model_name"],
            color=colors[i % len(colors)],
            alpha=0.87, edgecolor="white"
        )
        # Value labels on top of bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold"
            )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Model Performance Comparison", fontweight="bold", pad=14)
    ax.legend(framealpha=0.9)
    ax.axhline(0.8, color="gray", linestyle="--",
               alpha=0.4, linewidth=1, label="0.8 target")
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    if save:
        plt.savefig("data/plots/metrics_comparison.png",
                    dpi=150, bbox_inches="tight")
        print("Saved: data/plots/metrics_comparison.png")
    plt.show()


# ──────────────────────────────────────────────────
# 4. THRESHOLD ANALYSIS PLOT
# ──────────────────────────────────────────────────

def plot_threshold_analysis(model,
                             X_test: np.ndarray,
                             y_test: np.ndarray,
                             save: bool = True) -> None:
    """
    Show how Precision, Recall, and F1 change as we
    move the decision threshold from 0 to 1.

    This helps loan officers understand the trade-off:
    - Lower threshold → catch more defaults (higher recall)
      but also reject more good borrowers (lower precision)
    - Higher threshold → fewer false alarms
      but miss more real defaults
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.05, 0.95, 0.01)

    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
        f1s.append(f1_score(y_test, y_pred_t, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(thresholds, precisions, "#534AB7", linewidth=2,
            label="Precision")
    ax.plot(thresholds, recalls,    "#D85A30", linewidth=2,
            label="Recall (Default)")
    ax.plot(thresholds, f1s,        "#1D9E75", linewidth=2,
            label="F1 Score")

    # Mark default threshold
    ax.axvline(0.5, color="gray", linestyle="--",
               alpha=0.6, linewidth=1, label="Default threshold (0.5)")

    # Mark optimal threshold (best F1)
    best_idx = np.argmax(f1s)
    ax.axvline(thresholds[best_idx], color="#D4537E",
               linestyle=":", linewidth=2,
               label=f"Optimal F1 threshold ({thresholds[best_idx]:.2f})")

    ax.set_xlabel("Decision Threshold", fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Precision / Recall / F1 vs Decision Threshold",
                 fontweight="bold", pad=14)
    ax.legend(loc="center left", framealpha=0.9)
    ax.set_xlim([0.05, 0.95])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.2)

    plt.tight_layout()
    if save:
        plt.savefig("data/plots/threshold_analysis.png",
                    dpi=150, bbox_inches="tight")
        print("Saved: data/plots/threshold_analysis.png")
    plt.show()