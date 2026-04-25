FROM python:3.9-slim

RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ .
RUN mkdir -p model/artifacts
COPY model/artifacts/xgboost.joblib model/artifacts/xgboost.joblib
COPY model/artifacts/scaler.joblib model/artifacts/scaler.joblib

RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
