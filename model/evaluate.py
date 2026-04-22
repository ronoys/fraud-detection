"""
Evaluate the trained fraud detection model and print a classification report.

Usage:
    python evaluate.py --data path/to/creditcard.csv --model artifacts/model.joblib
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fraud detection model")
    parser.add_argument("--data", required=True, help="Path to creditcard.csv")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).parent / "artifacts" / "model.joblib"),
        help="Path to saved model artifact",
    )
    args = parser.parse_args()

    logger.info("Loading data from %s", args.data)
    df = pd.read_csv(args.data)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    logger.info("Loading model from %s", args.model)
    model = joblib.load(args.model)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print("\n=== Classification Report ===")
    print(classification_report(y, y_pred, target_names=["Legit", "Fraud"]))

    print("=== Additional Metrics ===")
    print(f"ROC-AUC  : {roc_auc_score(y, y_prob):.4f}")
    print(f"AUPRC    : {average_precision_score(y, y_prob):.4f}")

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")


if __name__ == "__main__":
    main()
