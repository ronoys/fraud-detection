import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from predictor import load_model, predict
from schemas import AlertEvent, PredictionResponse, TransactionRequest

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
        is_fraud, confidence = predict(transaction.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    response = PredictionResponse(
        fraud=is_fraud,
        confidence=confidence,
        transaction_id=transaction_id,
        timestamp=timestamp,
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
