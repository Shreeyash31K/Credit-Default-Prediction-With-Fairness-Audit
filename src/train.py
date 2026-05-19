"""
src/train.py
------------
Model training module.
Covers Logistic Regression (baseline) and XGBoost (main model)
with hyperparameter tuning, full evaluation, and model saving.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier


# ──────────────────────────────────────────────────
# 1. LOGISTIC REGRESSION — BASELINE
# ──────────────────────────────────────────────────

def train_logistic_regression(X_train: np.ndarray,
                               y_train: np.ndarray,
                               random_state: int = 42) -> LogisticRegression:
    """
    Train a Logistic Regression baseline model.

    WHY Logistic Regression as baseline?
    - Simple, fast, interpretable
    - Sets a performance floor — if XGBoost can't beat this,
      something is wrong with our data or pipeline
    - Well-calibrated probabilities out of the box

    Key parameters:
    - class_weight='balanced' : handles class imbalance 
      Equivalent to telling the model: "penalise missing a
      default more than penalising a false alarm."
    - max_iter=1000: ensure convergence 
    - C=1.0: inverse of regularisation strength (1.0 = moderate)
    """
    logger.info("Training Logistic Regression baseline...")

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",   # handles residual imbalance after SMOTE
        solver="lbfgs",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    logger.info("Logistic Regression training complete ✓")

    return model


# ──────────────────────────────────────────────────
# 2. XGBOOST — MAIN MODEL
# ──────────────────────────────────────────────────

def train_xgboost(X_train: np.ndarray,
                  y_train: np.ndarray,
                  random_state: int = 42) -> XGBClassifier:
    """
    Train an XGBoost classifier with sensible default parameters.

    WHY XGBoost?
    - Handles tabular data extremely well
    - Built-in regularisation (prevents overfitting)
    - TreeSHAP integration for fast explainability
    - Handles class imbalance via scale_pos_weight

    Key parameters explained:
    - n_estimators=300: number of trees to build
    - max_depth=4: how deep each tree can go (shallower = less overfit)
    - learning_rate=0.05: step size for each tree (smaller = more robust)
    - subsample=0.8: use 80% of rows per tree (prevents overfit)
    - colsample_bytree=0.8: use 80% of features per tree
    - scale_pos_weight: compensate for any remaining imbalance
      = count(negative class) / count(positive class)
    - eval_metric='auc': optimise for ROC-AUC during training
    - early_stopping_rounds: stop if no improvement for N rounds
    """
    logger.info("Training XGBoost model...")

    # Calculate scale_pos_weight for any remaining imbalance
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    scale_pos_weight = neg / pos
    logger.info(f"scale_pos_weight = {scale_pos_weight:.3f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        verbose=False,
    )

    logger.info(f"XGBoost training complete ✓ | Trees: {model.n_estimators}")
    return model


# ──────────────────────────────────────────────────
# 3. HYPERPARAMETER TUNING — GridSearchCV
# ──────────────────────────────────────────────────

def tune_xgboost(X_train: np.ndarray,
                 y_train: np.ndarray,
                 random_state: int = 42) -> XGBClassifier:
    """
    Tune XGBoost hyperparameters using GridSearchCV with
    Stratified K-Fold cross-validation.

    WHY GridSearchCV?
    - Exhaustively tests every combination of parameters
    - Uses cross-validation to estimate true generalisation
    - Returns the best model automatically

    WHY StratifiedKFold?
    - Ensures each fold has the same class ratio
    - Gives reliable metric estimates even with class imbalance

    WHY scoring='roc_auc'?
    - AUC is threshold-independent
    - Measures ability to rank defaulters above non-defaulters
    - Best metric when class distribution is skewed
    """
    logger.info("Starting XGBoost hyperparameter tuning (GridSearchCV)...")
    logger.info("This may take 2-5 minutes...")

    param_grid = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [3, 4, 6],
        "learning_rate":    [0.01, 0.05, 0.1],
        "subsample":        [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
    }

    # Total combinations: 3×3×3×2×2 = 108 fits × 5 folds = 540 model fits
    # For faster tuning during development, use smaller grid (see below)

    # --- FAST VERSION (use during development) ---
    param_grid_fast = {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 4],
        "learning_rate": [0.05, 0.1],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    base_model = XGBClassifier(
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid_fast,   # swap to param_grid for full search
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,                    # refit best model on full training set
    )

    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


# ──────────────────────────────────────────────────
# 4. EVALUATE MODEL
# ──────────────────────────────────────────────────

def evaluate_model(model,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   model_name: str = "Model") -> dict:
    """
    Full evaluation of a trained model on the test set.

    Returns a dict with all metrics for easy comparison
    and downstream reporting.

    Understanding the threshold:
    - model.predict() uses 0.5 as default threshold
    - For credit risk, we may want a LOWER threshold (e.g. 0.3)
      to catch more defaults (increase recall) at the cost of
      more false alarms (lower precision)
    - We evaluate at 0.5 here, then show threshold tuning below
    """
    logger.info(f"Evaluating {model_name}...")

    # Predictions
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]   # probability of default

    # Core metrics
    metrics = {
        "model_name":  model_name,
        "accuracy":    accuracy_score(y_test, y_pred),
        "precision":   precision_score(y_test, y_pred, zero_division=0),
        "recall":      recall_score(y_test, y_pred, zero_division=0),
        "f1":          f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":     roc_auc_score(y_test, y_prob),
        "conf_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # Full report
    report = classification_report(
        y_test, y_pred,
        target_names=["No Default", "Default"],
        output_dict=True
    )
    metrics["classification_report"] = report

    # ROC curve data (for plotting)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    metrics["roc_curve"] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }

    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info(f"  {model_name} — Evaluation Results")
    logger.info(f"{'='*50}")
    logger.info(f"  Accuracy  : {metrics['accuracy']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"  F1 Score  : {metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    logger.info(f"{'='*50}")

    return metrics


# ──────────────────────────────────────────────────
# 5. THRESHOLD TUNING
# ──────────────────────────────────────────────────

def find_optimal_threshold(model,
                            X_test: np.ndarray,
                            y_test: np.ndarray,
                            target_recall: float = 0.75) -> float:
    """
    Find the decision threshold that achieves a target recall
    while maximising precision.

    WHY threshold tuning matters in credit risk:
    - Default threshold (0.5) treats FP and FN equally
    - In lending, missing a defaulter (FN) is more expensive
      than a false alarm (FP) — you lose the loan amount
    - Lowering threshold to 0.3-0.4 catches more defaulters
      at the cost of rejecting some good borrowers

    Strategy:
    - We want recall >= target_recall for the Default class
    - Among all thresholds meeting that, pick the one with
      the best F1 score
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    best_threshold = 0.5
    best_f1 = 0.0

    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred_t = (y_prob >= threshold).astype(int)
        recall_t = recall_score(y_test, y_pred_t, zero_division=0)
        f1_t     = f1_score(y_test, y_pred_t, zero_division=0)

        if recall_t >= target_recall and f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = threshold

    logger.info(f"Optimal threshold: {best_threshold:.2f} "
                f"(target recall >= {target_recall}, best F1 = {best_f1:.4f})")

    return best_threshold


# ──────────────────────────────────────────────────
# 6. SAVE MODEL + METADATA
# ──────────────────────────────────────────────────

def save_model(model,
               metrics: dict,
               model_name: str = "xgboost",
               version: str = "v1",
               output_dir: str = "models") -> str:
    """
    Save the trained model and its metadata.

    Saves:
    - <model_name>_<version>.joblib  → the sklearn/xgboost model
    - <model_name>_<version>_meta.json → metrics + params + timestamp

    Versioning strategy:
    - v1, v2, v3 ... incremented manually per experiment
    - In production, use MLflow or DVC for automatic versioning
    """
    os.makedirs(output_dir, exist_ok=True)

    # Model file
    model_path = os.path.join(output_dir, f"{model_name}_{version}.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved: {model_path}")

    # Metadata (metrics + params + timestamp)
    meta = {
        "model_name":  model_name,
        "version":     version,
        "timestamp":   datetime.now().isoformat(),
        "metrics": {
            "accuracy":  round(metrics["accuracy"], 4),
            "precision": round(metrics["precision"], 4),
            "recall":    round(metrics["recall"], 4),
            "f1":        round(metrics["f1"], 4),
            "roc_auc":   round(metrics["roc_auc"], 4),
        },
        "params": (
            model.get_params()
            if hasattr(model, "get_params") else {}
        ),
    }

    meta_path = os.path.join(output_dir, f"{model_name}_{version}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Metadata saved: {meta_path}")
    return model_path