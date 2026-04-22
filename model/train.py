"""
Train an XGBoost fraud detection model on the Kaggle creditcard dataset.

Usage:
    python train.py --data path/to/creditcard.csv

Output:
    model/artifacts/model.joblib
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def load_data(csv_path: str) -> tuple:
    logger.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    scale_pos_weight=577,  # ~ratio of non-fraud to fraud in Kaggle dataset
                    use_label_encoder=False,
                    eval_metric="aucpr",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--data", required=True, help="Path to creditcard.csv")
    args = parser.parse_args()

    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info("Training model on %d samples...", len(X_train))
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    output_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(pipeline, output_path)
    logger.info("Model saved to %s", output_path)

    # Basic evaluation logged to console; see evaluate.py for full metrics
    from sklearn.metrics import average_precision_score
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auprc = average_precision_score(y_test, y_prob)
    logger.info("Validation AUPRC: %.4f", auprc)


if __name__ == "__main__":
    main()
