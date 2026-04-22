"""
Evaluate the best trained fraud detection model.

Usage:
    python evaluate.py

Output (saved to model/artifacts/evaluation/):
    roc_curve.png
    pr_curve.png
    confusion_matrix.png
    classification_report.txt
"""

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

import kagglehub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
EVAL_DIR = ARTIFACTS_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts():
    model = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    logger.info("Loaded best_model.joblib and scaler.joblib")
    return model, scaler


def load_test_data(scaler):
    """Reproduce the exact same test split used during training."""
    logger.info("Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = Path(path) / "creditcard.csv"
    df = pd.read_csv(csv_path)

    df = df.drop(columns=["Time"])
    X = df.drop(columns=["Class"])
    y = df["Class"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_test = X_test.copy()
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])
    logger.info("Test set: %d samples  (Legit: %d  Fraud: %d)",
                len(y_test), (y_test == 0).sum(), (y_test == 1).sum())
    return X_test, y_test


def plot_roc_curve(y_test, y_prob) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, color="#2563eb", label=f"ROC AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Fraud Detection", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = EVAL_DIR / "roc_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_pr_curve(y_test, y_prob) -> None:
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    baseline = (y_test == 1).mean()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, lw=2, color="#16a34a", label=f"AP = {ap:.4f}")
    ax.axhline(y=baseline, color="k", linestyle="--", lw=1,
               label=f"Random baseline ({baseline:.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Fraud Detection", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = EVAL_DIR / "pr_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


def plot_confusion_matrix(y_test, y_pred) -> None:
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Legit", "Fraud"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix — Fraud Detection", fontsize=14)
    fig.tight_layout()
    out = EVAL_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)


def threshold_analysis(y_test, y_prob) -> None:
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    col_w = 12
    header = (
        f"{'Threshold':>{col_w}}"
        f"{'Precision':>{col_w}}"
        f"{'Recall':>{col_w}}"
        f"{'F1':>{col_w}}"
    )
    print("\n=== Threshold Analysis ===")
    print(header)
    print("-" * len(header))
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        p = precision_score(y_test, y_pred_t, zero_division=0)
        r = recall_score(y_test, y_pred_t, zero_division=0)
        f = f1_score(y_test, y_pred_t, zero_division=0)
        print(
            f"{t:>{col_w}.1f}"
            f"{p:>{col_w}.4f}"
            f"{r:>{col_w}.4f}"
            f"{f:>{col_w}.4f}"
        )


def main():
    model, scaler = load_artifacts()
    X_test, y_test = load_test_data(scaler)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    plot_roc_curve(y_test, y_prob)
    plot_pr_curve(y_test, y_prob)
    plot_confusion_matrix(y_test, y_pred)

    report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
    report_path = EVAL_DIR / "classification_report.txt"
    report_path.write_text(report)
    logger.info("Saved %s", report_path)

    print("\n=== Classification Report ===")
    print(report)
    print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC  : {average_precision_score(y_test, y_prob):.4f}")

    threshold_analysis(y_test, y_prob)

    print(f"\nAll outputs saved to {EVAL_DIR}")


if __name__ == "__main__":
    main()
