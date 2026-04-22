# Fraud Detection System

A production-style real-time credit card fraud detection pipeline built for a university MLOps course. It demonstrates end-to-end ML engineering: model training, REST API serving, containerization, AWS alerting, observability, and load testing.

---

## Architecture Overview

```
Client Request
      │
      ▼
 FastAPI (/predict)
      │
      ├── Loads XGBoost model from disk at startup
      ├── Runs inference on transaction features (V1-V28, Time, Amount)
      ├── If fraud detected → publishes alert to AWS SNS Topic
      │                             │
      │                             └── SNS → SQS Queue (downstream consumers)
      └── Returns { fraud, confidence, transaction_id, timestamp }

Observability:
  - Prometheus scrapes /metrics (via prometheus-fastapi-instrumentator)
  - Grafana visualizes request latency, fraud rate, throughput

Load Testing:
  - k6 ramps to 50 VUs and validates p95 latency < 500ms
```

**Tech stack:** Python 3.11, FastAPI, XGBoost, scikit-learn, Docker, AWS EC2/SNS/SQS, Prometheus, Grafana, k6

---

## Setup Instructions

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- AWS account (for SNS/SQS alerts)
- k6 (for load testing)

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/fraud-detection-system.git
cd fraud-detection-system
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your AWS credentials and resource ARNs
```

### 3. Train the model

```bash
cd model
pip install scikit-learn xgboost joblib pandas
python train.py
```

This saves the trained model to `model/artifacts/model.joblib`.

### 4. Build and start services

```bash
docker-compose -f infra/docker-compose.yml up --build
```

---

## Running Locally

### Start the full stack (API + Prometheus + Grafana)

```bash
docker-compose -f infra/docker-compose.yml up --build
```

| Service    | URL                        |
|------------|----------------------------|
| API        | http://localhost:8000       |
| API Docs   | http://localhost:8000/docs  |
| Prometheus | http://localhost:9090       |
| Grafana    | http://localhost:3000       |

### Health check

```bash
curl http://localhost:8000/health
```

### Example predict request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 1000, "V1": -1.36, "V2": -0.07, ..., "V28": 0.02, "Amount": 149.62}'
```

### Run load tests

```bash
k6 run load-testing/script.js
```

---

## Project Structure

```
fraud-detection-system/
├── .github/workflows/   # CI pipeline
├── model/               # Training and evaluation scripts
├── api/                 # FastAPI application
├── infra/               # Dockerfile, docker-compose, k8s manifests
├── load-testing/        # k6 load test scripts
├── monitoring/          # Prometheus and Grafana config
└── notebooks/           # EDA and model selection
```
