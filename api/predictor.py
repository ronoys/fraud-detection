import logging
import os
import joblib
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

# Feature order must match training data column order. The training pipeline drops
# Time and scales Amount before fitting the model.
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

_model = None
_scaler = None


def load_model() -> None:
    """Load the trained model from disk into the module-level singleton."""
    global _model, _scaler
    model_path = Path(os.getenv("MODEL_PATH", "model/artifacts/xgboost.joblib"))
    scaler_path = Path(os.getenv("SCALER_PATH", "model/artifacts/scaler.joblib"))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Run model/train.py first to generate it."
        )
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler artifact not found at '{scaler_path}'. "
            "Run model/train.py first to generate it."
        )
    _model = joblib.load(model_path)
    _scaler = joblib.load(scaler_path)
    logger.info("Model loaded from %s", model_path)
    logger.info("Scaler loaded from %s", scaler_path)
    warmup_payload = {"Amount": 1.0, **{f"V{i}": 0.0 for i in range(1, 29)}}
    predict(warmup_payload)
    logger.info("Model warmup complete")


def predict(features: dict) -> tuple[bool, float]:
    """
    Run inference on a single transaction.

    Args:
        features: dict mapping feature names to float values.

    Returns:
        (is_fraud, confidence) where confidence is the fraud class probability.
    """
    if _model is None or _scaler is None:
        raise RuntimeError("Model has not been loaded. Call load_model() first.")

    amount_frame = pd.DataFrame({"Amount": [features["Amount"]]})
    scaled_amount = float(_scaler.transform(amount_frame)[0][0])
    model_features = {**features, "Amount": scaled_amount}
    X = pd.DataFrame([[model_features[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    fraud_prob: float = float(_model.predict_proba(X)[0][1])
    threshold = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
    is_fraud = fraud_prob >= threshold
    return is_fraud, fraud_prob
