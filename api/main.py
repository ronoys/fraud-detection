import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from predictor import load_model, predict
from schemas import AlertEvent, NotifyRequest, PipelineStep, PredictionResponse, TransactionRequest

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

_alerts: list[AlertEvent] = []


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}


@app.get("/alerts", response_model=list[AlertEvent], tags=["ops"])
def alerts():
    return _alerts[-25:]


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict_fraud(transaction: TransactionRequest) -> PredictionResponse:
    transaction_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        is_fraud, confidence, scaled_amount = predict(transaction.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    threshold = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
    pipeline_steps = [
        PipelineStep(
            name="Transaction received",
            detail=f"Amount=${transaction.Amount:.2f}, 28 PCA features + time",
        ),
        PipelineStep(
            name="Amount normalized",
            detail=f"${transaction.Amount:.2f} → {scaled_amount:.4f} (StandardScaler)",
        ),
        PipelineStep(
            name="XGBoost inference",
            detail=f"predict_proba returned fraud probability = {confidence:.4f}",
        ),
        PipelineStep(
            name="Threshold applied",
            detail=f"{confidence:.4f} {'≥' if is_fraud else '<'} {threshold} → {'FRAUD' if is_fraud else 'LEGITIMATE'}",
        ),
    ]

    response = PredictionResponse(
        fraud=is_fraud,
        confidence=confidence,
        transaction_id=transaction_id,
        timestamp=timestamp,
        pipeline_steps=pipeline_steps,
    )

    if is_fraud:
        alert = AlertEvent(
            transaction_id=transaction_id,
            amount=transaction.Amount,
            confidence=confidence,
            timestamp=timestamp,
        )
        _alerts.append(alert)
        logger.warning(
            "Fraud alert transaction_id=%s amount=%.2f confidence=%.4f",
            transaction_id,
            transaction.Amount,
            confidence,
        )

    return response


@app.post("/notify", tags=["ops"])
def notify(request: NotifyRequest):
    phone = os.getenv("ALERT_PHONE_NUMBER", "+18489994153")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    dt = datetime.fromisoformat(request.timestamp.replace("Z", "+00:00"))
    formatted_time = dt.strftime("%B %d, %Y at %I:%M %p UTC")
    message = f"Was this charge for ${request.amount:.2f} on {formatted_time} you? Reply STOP to opt out."
    try:
        sns = boto3.client("sns", region_name=region)
        sns.publish(PhoneNumber=phone, Message=message)
        logger.info("SMS alert sent for transaction_id=%s", request.transaction_id)
        return {"sent": True, "message": message}
    except (BotoCoreError, ClientError) as exc:
        logger.error("SNS publish failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
