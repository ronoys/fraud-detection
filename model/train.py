"""
Train four fraud detection models on the Kaggle Credit Card Fraud dataset.

Usage:
    python train.py

Output:
    model/artifacts/<model_name>.joblib  — each trained model
    model/artifacts/best_model.joblib    — best model by PR-AUC
    model/artifacts/scaler.joblib        — fitted StandardScaler for Amount
    model/artifacts/model_metadata.json  — metrics and metadata
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import kagglehub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    logger.info("Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = Path(path) / "creditcard.csv"
    logger.info("Loading CSV from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Dataset shape: %s", df.shape)
    counts = df["Class"].value_counts()
    logger.info("Class distribution — Legit: %d  Fraud: %d", counts[0], counts[1])
    return df


def prepare_split(df: pd.DataFrame):
    df = df.drop(columns=["Time"])
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info("Train: %d  Test: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def scale_amount(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit scaler on training Amount only; transform both splits."""
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])
    return X_train, X_test, scaler


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    logger.info("Applying SMOTE to training data...")
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)
    logger.info(
        "After SMOTE — size: %d  (Legit: %d  Fraud: %d)",
        len(y_sm), (y_sm == 0).sum(), (y_sm == 1).sum(),
    )
    return X_sm, y_sm


def build_models(neg: int, pos: int) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            C=0.01, max_iter=1000, random_state=42
        ),
        "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "xgboost": XGBClassifier(
            scale_pos_weight=neg / pos,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        ),
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    X_arr = X_test.to_numpy()
    y_pred = model.predict(X_arr)
    y_prob = model.predict_proba(X_arr)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    return {
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }


def print_table(results: dict) -> None:
    cols = ["roc_auc", "pr_auc", "f1", "precision", "recall", "training_time_seconds"]
    col_w = 16
    name_w = 24
    header = f"{'Model':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in cols)
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for name, r in results.items():
        row = f"{name:<{name_w}}"
        for c in cols:
            val = r.get(c, 0)
            row += f"{val:>{col_w}.4f}" if c != "training_time_seconds" else f"{val:>{col_w - 1}.1f}s"
        print(row)
    print(sep)


def main():
    df = load_data()
    dataset_shape = list(df.shape)
    class_dist = {str(k): int(v) for k, v in df["Class"].value_counts().items()}

    X_train, X_test, y_train, y_test = prepare_split(df)

    neg_train = int((y_train == 0).sum())
    pos_train = int((y_train == 1).sum())
    logger.info(
        "Original train class counts — Legit: %d  Fraud: %d  (scale_pos_weight=%.1f)",
        neg_train, pos_train, neg_train / pos_train,
    )

    X_train_scaled, X_test_scaled, scaler = scale_amount(X_train, X_test)
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    logger.info("Scaler saved to artifacts/scaler.joblib")

    X_sm, y_sm = apply_smote(X_train_scaled, y_train)

    models = build_models(neg_train, pos_train)
    all_results = {}

    for name, model in models.items():
        logger.info("--- Training: %s ---", name)
        if name == "knn":
            logger.info("KNN stores all training points — prediction will be slow on this dataset size")

        t0 = time.time()
        model.fit(X_sm, y_sm)
        duration = round(time.time() - t0, 2)
        logger.info("%s trained in %.1fs", name, duration)

        logger.info("Evaluating %s on test set...", name)
        metrics = evaluate_model(model, X_test_scaled, y_test)
        metrics["training_time_seconds"] = duration
        all_results[name] = metrics

        joblib.dump(model, ARTIFACTS_DIR / f"{name}.joblib")
        logger.info(
            "%s → PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f  Precision=%.4f  Recall=%.4f",
            name, metrics["pr_auc"], metrics["roc_auc"],
            metrics["f1"], metrics["precision"], metrics["recall"],
        )

    best_name = max(all_results, key=lambda n: all_results[n]["pr_auc"])
    joblib.dump(models[best_name], ARTIFACTS_DIR / "best_model.joblib")

    metadata = {
        "best_model_name": best_name,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_shape": dataset_shape,
        "class_distribution": class_dist,
        "models": all_results,
    }
    with open(ARTIFACTS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to artifacts/model_metadata.json")

    print_table(all_results)

    best = all_results[best_name]
    print(f"\nSelected : {best_name}")
    print(
        f"Reason   : Highest PR-AUC ({best['pr_auc']:.4f}) — preferred metric for severely imbalanced classes"
    )
    print(f"Saved to : model/artifacts/best_model.joblib")


if __name__ == "__main__":
    main()
