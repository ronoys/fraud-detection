"""
Extract correctly-classified fraud and legitimate examples from the test set.
Prints them as Python dicts ready to paste into frontend/app.py.

Usage:
    python model/extract_examples.py
"""

from pathlib import Path

import joblib
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def main():
    print("Loading model and scaler...")
    model = joblib.load(ARTIFACTS_DIR / "xgboost.joblib")
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")

    print("Downloading dataset...")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    df = pd.read_csv(Path(path) / "creditcard.csv")

    # Reproduce exact train/test split from training
    X = df.drop(columns=["Class"])
    y = df["Class"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    X_test = X_test.copy()
    time_col = X_test["Time"].copy()
    X_test_scaled = X_test.copy()
    X_test_scaled["Amount"] = scaler.transform(X_test[["Amount"]])
    X_test_scaled = X_test_scaled.drop(columns=["Time"])

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    v_cols = [f"V{i}" for i in range(1, 29)]

    results = X_test.copy()
    results["_time"] = time_col
    results["_true"] = y_test.values
    results["_pred"] = y_pred
    results["_prob"] = y_prob

    # Correctly-classified fraud: highest confidence first
    fraud_hits = results[(results["_true"] == 1) & (results["_pred"] == 1)].sort_values("_prob", ascending=False)
    # Correctly-classified legit: most confidently legitimate first
    legit_hits = results[(results["_true"] == 0) & (results["_pred"] == 0)].sort_values("_prob").head(20)
    # Pick legit examples with varied amounts for a better demo
    legit_hits = legit_hits.sample(3, random_state=7).sort_values("Amount")

    def row_to_dict(row, label):
        print(f'\n    "{label}": {{')
        print(f'        "amount": {round(row["Amount"], 2)},')
        print(f'        "time": {round(row["_time"], 1)},')
        for col in v_cols:
            print(f'        "{col}": {round(row[col], 4)},')
        print(f'        # fraud_prob={row["_prob"]:.4f}')
        print("    },")

    print("\n\n# ── Paste this EXAMPLES dict into frontend/app.py ──────────────────")
    print("EXAMPLES = {")

    for i, (_, row) in enumerate(legit_hits.iterrows(), 1):
        row_to_dict(row, f"Legitimate example {i}")

    for i, (_, row) in enumerate(fraud_hits.head(3).iterrows(), 1):
        row_to_dict(row, f"Fraudulent example {i}")

    print("}")
    print("\n# ─────────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
