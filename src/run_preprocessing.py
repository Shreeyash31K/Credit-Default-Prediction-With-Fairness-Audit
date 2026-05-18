"""
src/run_preprocessing.py
------------------------
Runs the full preprocessing pipeline end-to-end.
Saves preprocessor artifact to models/ for use in training & inference.
"""

import os
import sys
import joblib
from loguru import logger

# Configure logger
os.makedirs("logs", exist_ok=True)
logger.add("logs/preprocessing.log", rotation="1 MB", level="INFO")

sys.path.append("src")
from preprocess import (
    load_data,
    validate_data,
    engineer_features,
    build_preprocessor,
    split_data,
    apply_smote,
)


def main():

    logger.info("=" * 50)
    logger.info("PHASE 2: PREPROCESSING PIPELINE STARTED")
    logger.info("=" * 50)

    # 1. Load data
    df = load_data("data/GermanCredit.csv")

    # 2. Validate data
    validate_data(df)

    # 3. Feature engineering
    df = engineer_features(df)

    # 4. Split data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        protected_train,
        protected_test
    ) = split_data(df)

    # 5. Build preprocessor
    preprocessor = build_preprocessor()

    # 6. Transform data
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    logger.info(f"X_train shape: {X_train_t.shape}")
    logger.info(f"X_test shape: {X_test_t.shape}")

    # 7. Apply SMOTE
    X_train_balanced, y_train_balanced = apply_smote(
        X_train_t,
        y_train,
        strategy="smote"
    )

    # 8. Save artifacts
    os.makedirs("models", exist_ok=True)

    joblib.dump(preprocessor, "models/preprocessor_v1.joblib")
    joblib.dump(X_train_balanced, "models/X_train_balanced.joblib")
    joblib.dump(X_test_t, "models/X_test.joblib")
    joblib.dump(y_train_balanced, "models/y_train_balanced.joblib")
    joblib.dump(y_test, "models/y_test.joblib")
    joblib.dump(protected_train, "models/protected_train.joblib")
    joblib.dump(protected_test, "models/protected_test.joblib")

    logger.info("All artifacts saved to models/ ✓")
    logger.info("PHASE 2 COMPLETE")

    # 9. Summary
    print("\n" + "=" * 50)
    print("PHASE 2 PREPROCESSING SUMMARY")
    print("=" * 50)
    print(f"Train rows (after SMOTE): {len(X_train_balanced)}")
    print(f"Test rows (untouched): {len(X_test_t)}")
    print(f"Features per row: {X_train_balanced.shape[1]}")
    print("Preprocessor saved to: models/preprocessor_v1.joblib")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main() 
