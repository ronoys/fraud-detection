import os
import logging
import joblib
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Feature order must match training data column order
FEATURE_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

_model = None


def load_model() -> None:
    """Load the trained model from disk into the module-level singleton."""
    global _model
    model_path = Path(os.getenv("MODEL_PATH", "model/artifacts/model.joblib"))
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Run model/train.py first to generate it."
        )
    _model = joblib.load(model_path)
    logger.info("Model loaded from %s", model_path)


def predict(features: dict) -> tuple[bool, float]:
    """
    Run inference on a single transaction.

    Args:
        features: dict mapping feature names to float values.

    Returns:
        (is_fraud, confidence) where confidence is the fraud class probability.
    """
    if _model is None:
        raise RuntimeError("Model has not been loaded. Call load_model() first.")

    X = np.array([[features[col] for col in FEATURE_COLUMNS]])
    fraud_prob: float = float(_model.predict_proba(X)[0][1])
    threshold = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
    is_fraud = fraud_prob >= threshold
    return is_fraud, fraud_prob
