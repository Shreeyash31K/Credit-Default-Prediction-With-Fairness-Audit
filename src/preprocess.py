"""
src/preprocess.py
-----------------
Production preprocessing pipeline for the German Credit Dataset.
Handles encoding, scaling, splitting, and class imbalance.
"""

import pandas as pd
import numpy as np
import joblib
import os
from loguru import logger

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter


# ──────────────────────────────────────────────────
# 1. COLUMN DEFINITIONS
# ──────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "status",
    "credit_history",
    "purpose",
    "savings",
    "employment_duration",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

NUMERICAL_COLS = [
    "duration",
    "amount",
    "installment_rate",
    "present_residence",
    "age",
    "number_credits",
    "people_liable",
]

# Protected attributes (used in fairness audit later)
PROTECTED_ATTRS = ["age", "personal_status_sex"]

TARGET_COL = "credit_risk"


# ──────────────────────────────────────────────────
# 2. LOAD DATA
# ──────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load German Credit dataset and standardize column names.
    """

    logger.info(f"Loading dataset from: {filepath}")

    df = pd.read_csv(filepath)

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    logger.info(f"Dataset shape: {df.shape}")

    return df


# ──────────────────────────────────────────────────
# 3. VALIDATE DATA
# ──────────────────────────────────────────────────

def validate_data(df: pd.DataFrame) -> None:
    """
    Run basic data quality checks.
    Raises ValueError if critical issues are found.
    """
    logger.info("Running data validation checks...")

    # Check expected columns exist
    expected_cols = CATEGORICAL_COLS + NUMERICAL_COLS + [TARGET_COL]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Check missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"Missing values found:\n{missing[missing > 0]}")
    else:
        logger.info("No missing values found ✓")

    # Check target values
    unique_targets = set(df[TARGET_COL].unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Unexpected target values: {unique_targets}")

    logger.info("Data validation passed ✓")


# ──────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ──────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones.
    These are domain-driven features for credit risk.
    """
    df = df.copy()

    # Debt-to-income proxy:
    # Higher installment rate + longer duration = more burden
    df["debt_burden"] = df["installment_rate"] * df["duration"]

    # Age group (for fairness tracking, not direct model input)
    df["age_group"] = pd.cut(
        df["age"],
        bins=[18, 25, 35, 50, 75],
        labels=["18-25", "26-35", "36-50", "51+"]
    )

    # Credit amount per month
    df["monthly_credit_amount"] = (
        df["amount"] / df["duration"].replace(0, 1)
    )

    logger.info("Feature engineering complete. New features: "
                "debt_burden, age_group, monthly_credit_amount")

    return df


# ──────────────────────────────────────────────────
# 5. BUILD PREPROCESSING PIPELINE
# ──────────────────────────────────────────────────

def build_preprocessor() -> Pipeline:
    """
    Build a sklearn Pipeline that handles:
    - Ordinal encoding for categorical columns
    - Standard scaling for numerical columns

    Why OrdinalEncoder (not OneHotEncoder)?
    - XGBoost handles ordinal integers natively
    - OneHot creates 50+ columns → slower + overfitting risk
    - For LR we will use OneHot instead (see train.py)
    """
    from sklearn.compose import ColumnTransformer

    # Columns to use in model (exclude protected attrs from direct input)
    # We keep them for fairness audit but optionally remove from features
    feature_cats = CATEGORICAL_COLS
    feature_nums = NUMERICAL_COLS + ["debt_burden", "monthly_credit_amount"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ),
                feature_cats,
            ),
            (
                "num",
                StandardScaler(),
                feature_nums,
            ),
        ],
        remainder="drop",   # drop age_group (string), kept for fairness only
    )

    return preprocessor


# ──────────────────────────────────────────────────
# 6. TRAIN-TEST SPLIT
# ──────────────────────────────────────────────────

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train-test split.

    Why stratified?
    - Ensures both splits have the same 70/30 class ratio
    - Without stratification, random splits can skew the class ratio
    """
    feature_cols = (
        CATEGORICAL_COLS + NUMERICAL_COLS +
        ["debt_burden", "monthly_credit_amount"]
    )

    X = df[feature_cols]
    y = df[TARGET_COL]

    # Keep protected attributes separately for fairness audit
    protected = df[PROTECTED_ATTRS].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y           # ← CRITICAL: maintain class ratio
    )

    # Align protected attributes with the split
    protected_train = protected.loc[X_train.index]
    protected_test  = protected.loc[X_test.index]

    logger.info(f"Train size : {X_train.shape[0]} rows")
    logger.info(f"Test size  : {X_test.shape[0]} rows")
    logger.info(f"Train class dist: {Counter(y_train)}")
    logger.info(f"Test class dist : {Counter(y_test)}")

    return X_train, X_test, y_train, y_test, protected_train, protected_test

# ──────────────────────────────────────────────────
# 7. CLASS IMBALANCE — SMOTE
# ──────────────────────────────────────────────────

def apply_smote(X_train_transformed: np.ndarray,
                y_train: pd.Series,
                strategy: str = "smote",
                random_state: int = 42):
    """
    Apply oversampling to the TRAINING SET ONLY.

    Why only training set?
    - Applying SMOTE to test set would contaminate evaluation
    - Test set must reflect the REAL world distribution

    strategy options:
    - "smote"    : Generate synthetic minority samples (preferred)
    - "random"   : Simple random oversampling (faster, less sophisticated)
    - "none"     : No resampling (use class_weight in model instead)

    How SMOTE works step-by-step:
    1. Take a minority sample (a real defaulter)
    2. Find its K nearest neighbors (also defaulters)
    3. Pick a random neighbor
    4. Create a synthetic point BETWEEN the two samples
    5. Repeat until classes are balanced
    """
    logger.info(f"Class distribution BEFORE {strategy.upper()}: "
                f"{Counter(y_train)}")

    if strategy == "smote":
        sampler = SMOTE(
            sampling_strategy="auto",  # balance to majority class count
            k_neighbors=5,             # use 5 nearest neighbors
            random_state=random_state
        )
    elif strategy == "random":
        sampler = RandomOverSampler(
            sampling_strategy="auto",
            random_state=random_state
        )
    elif strategy == "none":
        logger.info("No resampling applied.")
        return X_train_transformed, y_train
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    X_resampled, y_resampled = sampler.fit_resample(
        X_train_transformed, y_train
    )

    logger.info(f"Class distribution AFTER {strategy.upper()}: "
                f"{Counter(y_resampled)}")
    logger.info(f"Rows added: {len(X_resampled) - len(X_train_transformed)}")

    return X_resampled, y_resampled  