"""
src/explain.py
--------------
SHAP explainability module for the credit default model.
Covers:
  - TreeSHAP explainer setup
  - Global importance (summary plot)
  - Local explanation (single borrower)
  - Waterfall + force plots
  - JSON-serialisable SHAP output (for the API)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import joblib
from loguru import logger

os.makedirs("data/plots", exist_ok=True)
shap.initjs()   # enables JS plots in notebooks


# ─────────────────────────────────────────────
# 1.  BUILD THE EXPLAINER
# ─────────────────────────────────────────────

def build_explainer(model, X_train_transformed: np.ndarray):
    """
    Create a TreeSHAP explainer for XGBoost.

    Why TreeSHAP (not KernelSHAP)?
    - TreeSHAP is an exact, O(TLD²) algorithm designed
      specifically for tree ensembles (XGBoost, LightGBM).
    - KernelSHAP approximates SHAP by sampling coalitions 
      much slower and slightly less accurate.
    - For our XGBoost model TreeSHAP is always preferred.

    The background dataset:
    - SHAP uses a background (reference) distribution to
      compute "what is the average prediction without this
      feature?"  that is the base rate we saw above.
    - We pass a summary of the training data (100 samples)
      as the background for efficiency.
    """
    logger.info("Building TreeSHAP explainer...")

    background = shap.sample(X_train_transformed, 100, random_state=42)

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="probability",   # SHAP values in probability space
    )

    logger.info("TreeSHAP explainer ready ✓")
    return explainer


# ─────────────────────────────────────────────
# 2.  COMPUTE SHAP VALUES
# ─────────────────────────────────────────────

def compute_shap_values(explainer, X: np.ndarray) -> shap.Explanation:
    """
    Compute SHAP values for a set of samples.

    Returns a shap.Explanation object which contains:
    - .values       : (n_samples, n_features) array of SHAP values
    - .base_values  : scalar base rate (E[f(x)])
    - .data         : the original feature values

    For a binary classifier predicting default probability:
    - Positive SHAP value → pushes prediction ABOVE base rate
    - Negative SHAP value → pushes prediction BELOW base rate
    - |SHAP value| → magnitude of influence
    """
    logger.info(f"Computing SHAP values for {X.shape[0]} samples...")
    shap_values = explainer(X)
    logger.info("SHAP values computed ✓")
    return shap_values


# ─────────────────────────────────────────────
# 3.  GET FEATURE NAMES
# ─────────────────────────────────────────────

def get_feature_names(preprocessor) -> list:
    """
    Reconstruct feature names in the same order the
    ColumnTransformer outputs them: [cat features] + [num features]
    """
    cat_features = preprocessor.named_transformers_["cat"]\
                               .get_feature_names_out().tolist()
    num_features = [
        "duration", "credit_amount", "installment_rate",
        "residence_since", "age", "existing_credits",
        "liable_people", "debt_burden", "monthly_credit_amount"
    ]
    return cat_features + num_features


# ─────────────────────────────────────────────
# 4.  GLOBAL — SUMMARY PLOT
# ─────────────────────────────────────────────

def plot_shap_summary(shap_values: shap.Explanation,
                      feature_names: list,
                      max_display: int = 15,
                      save: bool = True) -> None:
    """
    Global SHAP summary plot — shows all features ranked by
    mean |SHAP value| across the entire test set.

    How to read it:
    - Y axis: features ranked by importance (top = most important)
    - X axis: SHAP value (positive = increases default risk)
    - Colour: actual feature value (red = high, blue = low)
    - Each dot = one borrower

    Insight patterns to look for:
    - A feature with all red dots on the right side means
      high values of that feature consistently increase risk
    - Wide spread = feature has varying impact across borrowers
    - Tight cluster near 0 = feature matters little
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    shap.summary_plot(
        shap_values.values,
        shap_values.data,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
        color_bar=True,
        plot_size=None,
    )

    ax = plt.gca()
    ax.set_title("SHAP Summary Plot — Global Feature Importance",
                 fontweight="bold", pad=14)
    ax.set_xlabel("SHAP value (impact on default probability)", fontweight="bold")

    plt.tight_layout()
    if save:
        plt.savefig("data/plots/shap_summary.png", dpi=150,
                    bbox_inches="tight")
        logger.info("Saved: data/plots/shap_summary.png")
    plt.show()


# ─────────────────────────────────────────────
# 5.  GLOBAL — BAR PLOT (mean |SHAP|)
# ─────────────────────────────────────────────

def plot_shap_bar(shap_values: shap.Explanation,
                  feature_names: list,
                  top_n: int = 12,
                  save: bool = True) -> None:
    """
    Bar chart of mean absolute SHAP values.
    Simpler to communicate to non-technical stakeholders than the dot plot.

    mean(|SHAP|) = average magnitude of influence across all borrowers.
    This is the standard global feature importance metric for SHAP.
    """
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names[:len(mean_abs)],
        "importance": mean_abs
    }).sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        importance_df["feature"],
        importance_df["importance"],
        color="#534AB7", alpha=0.85, edgecolor="white"
    )

    for bar, val in zip(bars, importance_df["importance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    ax.set_xlabel("Mean |SHAP value| (avg impact on default probability)",
                  fontweight="bold")
    ax.set_title(f"Top {top_n} Features — Global SHAP Importance",
                 fontweight="bold", pad=14)
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    if save:
        plt.savefig("data/plots/shap_bar.png", dpi=150,
                    bbox_inches="tight")
        logger.info("Saved: data/plots/shap_bar.png")
    plt.show()


# ─────────────────────────────────────────────
# 6.  LOCAL — WATERFALL PLOT (single borrower)
# ─────────────────────────────────────────────

def plot_waterfall(shap_values: shap.Explanation,
                   feature_names: list,
                   sample_idx: int = 0,
                   save: bool = True) -> None:
    """
    Waterfall plot for one borrower.

    This is the MOST IMPORTANT plot for loan officers.
    It shows exactly WHY the model gave this specific score.

    How to read it:
    - Starts at E[f(x)] — the average prediction (base rate)
    - Each bar = how much this feature pushed the score up or down
    - Red bars = increase default risk
    - Blue/teal bars = decrease default risk
    - Ends at f(x) — the final prediction for this borrower

    This is what you show in the dashboard for every application.
    """
    single = shap.Explanation(
        values=shap_values.values[sample_idx],
        base_values=shap_values.base_values[sample_idx],
        data=shap_values.data[sample_idx],
        feature_names=feature_names[:shap_values.values.shape[1]],
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(single, max_display=12, show=False)

    plt.title(f"SHAP Waterfall — Borrower #{sample_idx}",
              fontweight="bold", pad=14)
    plt.tight_layout()

    if save:
        path = f"data/plots/shap_waterfall_{sample_idx}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.show()


# ─────────────────────────────────────────────
# 7.  LOCAL — FORCE PLOT (interactive HTML)
# ─────────────────────────────────────────────

def plot_force(shap_values: shap.Explanation,
               feature_names: list,
               sample_idx: int = 0,
               save: bool = True) -> None:
    """
    SHAP force plot — interactive HTML version.

    Shows the same information as waterfall but as a
    horizontal "force" pushing left (reduce risk) or
    right (increase risk) from the base value.

    In notebooks this renders as an interactive HTML widget.
    In the API we'll export it as an HTML string.
    """
    force = shap.force_plot(
        base_value=shap_values.base_values[sample_idx],
        shap_values=shap_values.values[sample_idx],
        features=shap_values.data[sample_idx],
        feature_names=feature_names[:shap_values.values.shape[1]],
        matplotlib=False,       # interactive JS version
        show=False,
    )

    if save:
        path = f"data/plots/shap_force_{sample_idx}.html"
        shap.save_html(path, force)
        logger.info(f"Saved: {path}")

    return force


# ─────────────────────────────────────────────
# 8.  DEPENDENCY PLOT
# ─────────────────────────────────────────────

def plot_dependence(shap_values: shap.Explanation,
                    feature_names: list,
                    feature: str = "status",
                    save: bool = True) -> None:
    """
    Dependence plot for a single feature.

    Shows how SHAP value changes as the feature value changes.
    Colour = a second interacting feature (auto-selected by SHAP).

    Useful for:
    - Understanding non-linear relationships
    - Detecting interaction effects between features
    - Communicating feature behaviour to regulators
    """
    
    if feature not in feature_names:
        raise ValueError(
        f"Feature '{feature}' not found.\n"
        f"Available features:\n{feature_names}"
    )

    feat_idx = feature_names.index(feature)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feat_idx,
        shap_values.values,
        shap_values.data,
        feature_names=feature_names[:shap_values.values.shape[1]],
        ax=ax,
        show=False,
    )

    ax.set_title(f"SHAP Dependence — {feature}", fontweight="bold", pad=14)
    ax.set_ylabel(f"SHAP value for {feature}", fontweight="bold")

    plt.tight_layout()
    if save:
        path = f"data/plots/shap_dependence_{feature}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.show()


# ─────────────────────────────────────────────
# 9.  API-READY LOCAL EXPLANATION
# ─────────────────────────────────────────────

def explain_single(model,
                   explainer,
                   X_single: np.ndarray,
                   feature_names: list,
                   top_n: int = 8) -> dict:
    """
    Generate a JSON-serialisable SHAP explanation for ONE borrower.
    This is what the /explain FastAPI endpoint returns.

    Returns:
    {
      "base_value": 0.30,
      "prediction": 0.78,
      "top_features": [
        {"feature": "checking_account", "shap_value": 0.22,
         "feature_value": "A11", "direction": "increases_risk"},
        ...
      ]
    }
    """
    shap_vals = explainer(X_single)

    values       = shap_vals.values[0]
    base_value   = float(shap_vals.base_values[0])
    feat_data    = shap_vals.data[0]
    prediction   = float(base_value + np.sum(values))

    feat_names_trimmed = feature_names[:len(values)]

    # Sort by absolute magnitude
    order = np.argsort(np.abs(values))[::-1][:top_n]

    top_features = []
    for i in order:
        top_features.append({
            "feature":       feat_names_trimmed[i],
            "shap_value":    round(float(values[i]), 4),
            "feature_value": str(feat_data[i]),
            "direction":     "increases_risk" if values[i] > 0
                             else "decreases_risk",
        })

    return {
        "base_value":  round(base_value, 4),
        "prediction":  round(prediction, 4),
        "top_features": top_features,
    }
