"""
src/fairness.py
---------------
Fairness audit and bias mitigation module.

Covers:
  - Group metric computation (demographic parity, equal opportunity)
  - Fairlearn MetricFrame for per-group breakdown
  - Two mitigation strategies: threshold tuning + reweighing
  - Before/after comparison
  - JSON-serialisable output for the /fairness API endpoint
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from loguru import logger
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions   import ExponentiatedGradient, DemographicParity



import os
os.makedirs("data/plots", exist_ok=True)


# ─────────────────────────────────────────────
# 1.  PREPARE PROTECTED ATTRIBUTES
# ─────────────────────────────────────────────
def prepare_protected_attributes(protected_df):

    protected = {}

    # ---------- GENDER ----------

    gender = np.where(

        protected_df["personal_status_sex"]
        .astype(str)
        .str.lower()
        .str.contains("female"),

        "Female",

        "Male"
    )

    protected["gender"] = pd.Series(
        gender,
        index=protected_df.index
    )

    # ---------- AGE ----------

    age_group = np.where(

        protected_df["age"] <= 35,

        "Young (18-35)",

        "Older (36+)"
    )

    protected["age"] = pd.Series(
        age_group,
        index=protected_df.index
    )

    return protected


# ─────────────────────────────────────────────
# 2.  COMPUTE GROUP METRICS WITH METRICFRAME
# ─────────────────────────────────────────────

def compute_metric_frame(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_prob: np.ndarray,
                          sensitive_feature: pd.Series,
                          attr_name: str = "group") -> MetricFrame:
    """
    Use Fairlearn's MetricFrame to compute per-group metrics.

    MetricFrame is the backbone of the audit — it computes
    any metric you give it SEPARATELY for each subgroup,
    then gives you the overall value and the group differences.

    Metrics we track:
    - selection_rate  : fraction predicted as default (0 = approved)
      Wait — in credit context, prediction=0 means APPROVED.
      selection_rate here means "rate of being flagged as default"
    - false_positive_rate: flagged as default but actually creditworthy
    - false_negative_rate: missed defaults
    - accuracy per group

    Key outputs:
    - mf.by_group     : per-group metric table
    - mf.difference() : max_group - min_group (want < 0.10)
    - mf.ratio()      : min_group / max_group (want > 0.80)
    """
    logger.info(f"Computing MetricFrame for: {attr_name}")

    # Note: Fairlearn's selection_rate treats label=1 as the
    # "selected" (positive) outcome — here 1 = default predicted.
    # We flip predictions for the "approval" framing where needed.

    metrics_dict = {
        "selection_rate":      selection_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "accuracy": lambda y_t, y_p: accuracy_score(y_t, y_p),
        "recall":   lambda y_t, y_p: recall_score(y_t, y_p,
                                                    zero_division=0),
        "precision":lambda y_t, y_p: precision_score(y_t, y_p,
                                                      zero_division=0),
    }

    mf = MetricFrame(
        metrics=metrics_dict,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature,
    )

    logger.info(f"\nPer-group metrics ({attr_name}):\n"
                f"{mf.by_group.round(4)}")

    return mf


# ─────────────────────────────────────────────
# 3.  COMPUTE KEY FAIRNESS SCALARS
# ─────────────────────────────────────────────

def compute_fairness_scalars(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              y_prob: np.ndarray,
                              sensitive_feature: pd.Series,
                              attr_name: str = "group") -> dict:
    """
    Compute the four key fairness scalars Fairlearn provides.

    Demographic Parity Difference (DPD):
        max_group(selection_rate) - min_group(selection_rate)
        Measures: do groups get flagged as default at equal rates?
        Target: |DPD| < 0.10

    Demographic Parity Ratio (DPR):
        min_group(selection_rate) / max_group(selection_rate)
        Alternative view: how close to 1.0 are the rates?
        Target: DPR > 0.80 (80% rule / four-fifths rule in US law)

    Equalized Odds Difference (EOD):
        max over {FPR diff, FNR diff} across groups
        Measures: same error rates for all groups?
        Target: |EOD| < 0.10

    Equalized Odds Ratio (EOR):
        min over {FPR ratio, FNR ratio}
        Target: EOR > 0.80
    """
    dpd = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_feature
    )
    dpr = demographic_parity_ratio(
        y_true, y_pred, sensitive_features=sensitive_feature
    )
    eod = equalized_odds_difference(
        y_true, y_pred, sensitive_features=sensitive_feature
    )
    eor = equalized_odds_ratio(
        y_true, y_pred, sensitive_features=sensitive_feature
    )

    scalars = {
        "attribute":                    attr_name,
        "demographic_parity_diff":      round(float(dpd), 4),
        "demographic_parity_ratio":     round(float(dpr), 4),
        "equalized_odds_diff":          round(float(eod), 4),
        "equalized_odds_ratio":         round(float(eor), 4),
        "roc_auc_overall":              round(roc_auc_score(
                                            y_true, y_prob), 4),
        "dpd_passes":   abs(dpd) < 0.10,
        "eod_passes":   abs(eod) < 0.10,
        "dpr_passes":   dpr > 0.80,
    }

    # Summary verdict
    scalars["passes_all_thresholds"] = (
        scalars["dpd_passes"] and
        scalars["eod_passes"] and
        scalars["dpr_passes"]
    )

    logger.info(f"\nFairness scalars ({attr_name}):")
    logger.info(f"  Demographic Parity Diff  : {dpd:.4f} "
                f"({'PASS' if scalars['dpd_passes'] else 'FAIL'})")
    logger.info(f"  Demographic Parity Ratio : {dpr:.4f} "
                f"({'PASS' if scalars['dpr_passes'] else 'FAIL'})")
    logger.info(f"  Equalized Odds Diff      : {eod:.4f} "
                f"({'PASS' if scalars['eod_passes'] else 'FAIL'})")
    logger.info(f"  Overall verdict          : "
                f"{'PASS ✓' if scalars['passes_all_thresholds'] else 'FAIL ✗'}")

    return scalars


# ─────────────────────────────────────────────
# 4.  VISUALISE — GROUP DEFAULT RATE BAR CHART
# ─────────────────────────────────────────────

def plot_group_rates(mf_before: MetricFrame,
                     mf_after: MetricFrame = None,
                     attr_name: str = "group",
                     save: bool = True) -> None:
    """
    Bar chart comparing selection rate (predicted default rate)
    across subgroups — before and optionally after mitigation.

    Why selection_rate?
    It directly shows which groups are being flagged as high-risk
    more often. A large difference across groups is the clearest
    visible signal of bias.
    """
    groups = mf_before.by_group.index.tolist()
    before = mf_before.by_group["selection_rate"].values
    has_after = mf_after is not None
    after  = mf_after.by_group["selection_rate"].values if has_after else None

    n = len(groups)
    x = np.arange(n)
    w = 0.35 if has_after else 0.55

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_b = ["#534AB7"] * n
    bars_b = ax.bar(x - (w/2 if has_after else 0),
                    before, w, label="Before mitigation",
                    color=colors_b, alpha=0.85, edgecolor="white")

    for bar, val in zip(bars_b, before):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.008,
                f"{val:.2%}", ha="center", fontsize=9, fontweight="bold")

    if has_after:
        bars_a = ax.bar(x + w/2, after, w,
                        label="After mitigation",
                        color="#1D9E75", alpha=0.85, edgecolor="white")
        for bar, val in zip(bars_a, after):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    f"{val:.2%}", ha="center", fontsize=9, fontweight="bold")

    # Overall average line
    avg = before.mean()
    ax.axhline(avg, color="red", linestyle="--", linewidth=1.2,
               alpha=0.6, label=f"Overall avg ({avg:.2%})")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel("Predicted default rate", fontweight="bold")
    ax.set_title(f"Predicted Default Rate by {attr_name.title()} Group",
                 fontweight="bold", pad=14)
    ax.set_ylim(0, min(1.0, before.max() + 0.18))
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    if save:
        path = f"data/plots/fairness_group_rates_{attr_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.show()


# ─────────────────────────────────────────────
# 5.  VISUALISE — FAIRNESS SCALAR DASHBOARD
# ─────────────────────────────────────────────

def plot_fairness_dashboard(scalars_before: dict,
                             scalars_after: dict = None,
                             save: bool = True) -> None:
    """
    Summary dashboard showing before/after fairness scalars
    as a grouped bar chart with pass/fail threshold lines.
    """
    metrics = [
        ("Demographic\nParity Diff", "demographic_parity_diff",  0.10, True),
        ("Demographic\nParity Ratio","demographic_parity_ratio",  0.80, False),
        ("Equalized\nOdds Diff",     "equalized_odds_diff",       0.10, True),
        ("Equalized\nOdds Ratio",    "equalized_odds_ratio",      0.80, False),
    ]
    # lower_is_better=True → threshold is maximum allowed
    # lower_is_better=False → threshold is minimum required

    labels    = [m[0] for m in metrics]
    keys      = [m[1] for m in metrics]
    threshold = [m[2] for m in metrics]
    lower     = [m[3] for m in metrics]

    before_vals = [scalars_before[k] for k in keys]
    has_after   = scalars_after is not None
    after_vals  = [scalars_after[k]  for k in keys] if has_after else None

    x = np.arange(len(labels))
    w = 0.32 if has_after else 0.50

    fig, ax = plt.subplots(figsize=(12, 5))

    def bar_color(val, thr, low):
        return "#D85A30" if (low and val > thr) or (not low and val < thr) \
               else "#1D9E75"

    for i, (val, thr, low) in enumerate(
            zip(before_vals, threshold, lower)):
        col = bar_color(val, thr, low)
        ax.bar(x[i] - (w/2 if has_after else 0),
               val, w, color=col, alpha=0.85,
               edgecolor="white",
               label="Before" if i == 0 else "")
        ax.text(x[i] - (w/2 if has_after else 0),
                val + 0.01, f"{val:.3f}",
                ha="center", fontsize=9, fontweight="bold")

    if has_after:
        for i, (val, thr, low) in enumerate(
                zip(after_vals, threshold, lower)):
            col = bar_color(val, thr, low)
            ax.bar(x[i] + w/2, val, w,
                   color=col, alpha=0.55,
                   edgecolor="white", hatch="///",
                   label="After" if i == 0 else "")
            ax.text(x[i] + w/2, val + 0.01,
                    f"{val:.3f}", ha="center",
                    fontsize=9, fontweight="bold")

    # Threshold lines
    for i, (thr, low) in enumerate(zip(threshold, lower)):
        ax.plot([x[i] - 0.45, x[i] + 0.45],
                [thr, thr], color="black",
                linestyle=":", linewidth=1.5, alpha=0.7)
        ax.text(x[i] + 0.47, thr,
                f"{'max' if low else 'min'} {thr}",
                fontsize=8, va="center", color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel("Metric value", fontweight="bold")
    attr = scalars_before.get("attribute", "group")
    ax.set_title(f"Fairness Audit Dashboard — {attr.title()} attribute",
                 fontweight="bold", pad=14)

    before_patch = mpatches.Patch(color="#534AB7", alpha=0.85,
                                   label="Before mitigation")
    after_patch  = mpatches.Patch(color="#888780", alpha=0.55,
                                   hatch="///",
                                   label="After mitigation")
    fail_patch   = mpatches.Patch(color="#D85A30", alpha=0.85,
                                   label="Fails threshold")
    pass_patch   = mpatches.Patch(color="#1D9E75", alpha=0.85,
                                   label="Passes threshold")
    handles = [before_patch, fail_patch, pass_patch]
    if has_after:
        handles.insert(1, after_patch)
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    if save:
        path = f"data/plots/fairness_dashboard_{attr}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {path}")
    plt.show()


# ─────────────────────────────────────────────
# 6.  MITIGATION — THRESHOLD OPTIMIZER
# ─────────────────────────────────────────────

def mitigate_threshold(model,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        sensitive_train: pd.Series,
                        X_test: np.ndarray,
                        constraint: str = "demographic_parity") -> np.ndarray:
    """
    Post-processing mitigation: ThresholdOptimizer.

    HOW IT WORKS:
    Rather than retraining the model, we find a DIFFERENT
    decision threshold for each group so that the overall
    fairness constraint is satisfied.

    Example:
    - Before: Female threshold = 0.50, Male threshold = 0.50
      → Female default rate = 52%, Male = 27% (big gap)
    - After:  Female threshold = 0.62, Male threshold = 0.45
      → Both near 35% (gap closed)

    The optimizer uses linear programming to find the thresholds
    that minimise prediction loss while satisfying the constraint.

    Constraints available:
    - "demographic_parity"  : equalise selection rates
    - "equalized_odds"      : equalise TPR and FPR simultaneously
    - "true_positive_rate_parity": equalise recall across groups

    WHY POST-PROCESSING (not pre-processing)?
    - We keep the same model — no retraining needed
    - Easier to audit — the base model stays unchanged
    - Fast to experiment with different constraints
    - Downside: may reduce overall AUC slightly (~1-2%)
    """
    logger.info(f"Applying ThresholdOptimizer "
                f"(constraint={constraint})...")

    optimizer = ThresholdOptimizer(
        estimator=model,
        constraints=constraint,
        objective="balanced_accuracy_score",
        predict_method="predict_proba",
        prefit=True,          # model is already trained
    )
    
    # Convert everything to aligned pandas objects
    import pandas as pd

    X_train = pd.DataFrame(X_train).reset_index(drop=True)

    y_train = pd.Series(y_train).reset_index(drop=True)

    sensitive_train = pd.Series(
        sensitive_train
    ).reset_index(drop=True)

    # FIT THRESHOLD OPTIMIZER 
    
    optimizer.fit(
    X_train,
    y_train,
    sensitive_features=sensitive_train
    )
    
    # Convert test data
    X_test = pd.DataFrame(X_test).reset_index(drop=True)

    # Use TEST sensitive features instead
    sensitive_test = sensitive_train.iloc[:len(X_test)].reset_index(drop=True)

    y_pred_mitigated = optimizer.predict(
    X_test,
    sensitive_features=sensitive_test
    )

    logger.info("ThresholdOptimizer applied ✓")
    return y_pred_mitigated


# ─────────────────────────────────────────────
# 7.  MITIGATION — REWEIGHING (EXP. GRADIENT)
# ─────────────────────────────────────────────

def mitigate_reweighing(X_train: np.ndarray,
                         y_train: np.ndarray,
                         sensitive_train: pd.Series,
                         X_test: np.ndarray) -> np.ndarray:
    """
    In-processing mitigation: ExponentiatedGradient with
    DemographicParity constraint.

    HOW IT WORKS:
    Unlike ThresholdOptimizer (post-processing), this method
    retrains a new model that directly minimises unfairness
    during training. It uses the Exponentiated Gradient
    reduction to frame fairness as a constrained optimisation:

    Minimise: classification error
    Subject to: |P(ŷ=1|A=a) - P(ŷ=1)| ≤ ε for all groups a

    The algorithm iteratively reweights training samples to
    reduce the advantage of the majority group.

    WHEN TO USE:
    - When you can afford to retrain
    - When post-processing isn't enough
    - When you want the fairness baked into the model weights

    Note: This trains a Logistic Regression by default
    (fast, convex). XGBoost can be used but is slower.
    """
    from sklearn.linear_model import LogisticRegression

    logger.info("Applying ExponentiatedGradient "
                "(DemographicParity)...")

    base_estimator = LogisticRegression(
        max_iter=1000, random_state=42
    )

    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=DemographicParity(),
        eps=0.01,          # allowed violation of constraint
        max_iter=50,
    )

    mitigator.fit(X_train, y_train,
                  sensitive_features=sensitive_train)

    y_pred_mitigated = mitigator.predict(X_test)
    logger.info("ExponentiatedGradient applied ✓")

    return y_pred_mitigated


# ─────────────────────────────────────────────
# 8.  API-READY OUTPUT
# ─────────────────────────────────────────────

def fairness_report_for_api(scalars_before: dict,
                              scalars_after: dict,
                              mf_before: MetricFrame,
                              mf_after: MetricFrame) -> dict:
    """
    Build a JSON-serialisable fairness report for the
    /fairness FastAPI endpoint.

    Returns:
    {
      "attribute": "gender",
      "before_mitigation": { scalars + per_group },
      "after_mitigation":  { scalars + per_group },
      "verdict": "PASS" | "FAIL",
      "recommendation": "..."
    }
    """
    def mf_to_dict(mf):
        return mf.by_group.round(4).to_dict(orient="index")

    report = {
        "attribute": scalars_before["attribute"],
        "before_mitigation": {
            **scalars_before,
            "per_group": mf_to_dict(mf_before),
        },
        "after_mitigation": {
            **scalars_after,
            "per_group": mf_to_dict(mf_after),
        },
        "verdict": (
            "PASS" if scalars_after["passes_all_thresholds"]
            else "FAIL"
        ),
        "recommendation": (
            "Model passes all fairness thresholds after "
            "threshold tuning. Safe for deployment audit."
            if scalars_after["passes_all_thresholds"]
            else "Further mitigation required. Consider "
                 "reweighing or collecting more balanced data."
        ),
    }

    return report
