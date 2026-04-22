import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import boto3
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from predictor import load_model, predict
from schemas import PredictionResponse, TransactionRequest

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    load_model()
    logger.info("Model ready.")
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection powered by XGBoost.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

# SNS client — lazily initialized so the app still starts without AWS creds locally
_sns = None


def _get_sns():
    global _sns
    if _sns is None:
        _sns = boto3.client("sns", region_name=os.getenv("AWS_REGION", "us-east-1"))
    return _sns


def _publish_fraud_alert(transaction_id: str, amount: float, confidence: float) -> None:
    topic_arn = os.getenv("SNS_TOPIC_ARN")
    if not topic_arn:
        logger.warning("SNS_TOPIC_ARN not set — skipping alert publish.")
        return
    try:
        _get_sns().publish(
            TopicArn=topic_arn,
            Subject="Fraud Alert",
            Message=(
                f"Fraud detected!\n"
                f"Transaction ID : {transaction_id}\n"
                f"Amount         : ${amount:.2f}\n"
                f"Confidence     : {confidence:.4f}"
            ),
            MessageAttributes={
                "event_type": {"DataType": "String", "StringValue": "fraud_alert"}
            },
        )
        logger.info("Fraud alert published for transaction %s", transaction_id)
    except Exception:
        logger.exception("Failed to publish SNS alert for transaction %s", transaction_id)


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict_fraud(transaction: TransactionRequest) -> PredictionResponse:
    transaction_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        is_fraud, confidence = predict(transaction.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if is_fraud:
        _publish_fraud_alert(transaction_id, transaction.Amount, confidence)

    return PredictionResponse(
        fraud=is_fraud,
        confidence=confidence,
        transaction_id=transaction_id,
        timestamp=timestamp,
    )
