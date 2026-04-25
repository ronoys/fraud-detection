# Fraud Detection System

A containerized real-time credit card fraud detection demo for an MLOps course. The project serves a trained fraud model through FastAPI, provides a small Streamlit dashboard, exposes Prometheus metrics, and uses k6 to measure latency and throughput under load.

## Project Focus

The goal is to evaluate more than offline model accuracy. This system shows how a fraud model can be packaged as a deployable inference service and evaluated with production-style metrics:

- Prediction quality: ROC-AUC, PR-AUC, F1, precision, recall
- Service latency: P50, P95, P99 response time
- Throughput: requests per second under simulated transaction traffic
- Reliability: health checks, error rate, container restart behavior
- Portability: local Docker Compose deployment plus AWS Elastic Beanstalk Docker deployment

## Architecture

```text
Streamlit Frontend
        |
        v
FastAPI /predict
        |
        v
Trained XGBoost Fraud Model + Scaler

Observability:
FastAPI /metrics -> Prometheus -> Grafana

Benchmarking:
k6 -> FastAPI /predict
```

## Tech Stack

- Python, scikit-learn, XGBoost, joblib
- FastAPI for model serving
- Streamlit for the demo frontend
- Docker Compose for local multi-container deployment
- Prometheus and Grafana for metrics
- k6 for load testing
- AWS Elastic Beanstalk for the cloud container deployment

## Local Setup

Create an environment file:

```bash
cp .env.example .env
```

Start the stack:

```bash
docker-compose -f infra/docker-compose.yml up --build
```

Services:

| Service | URL |
|---|---|
| Frontend | http://localhost:8501 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Grafana login:

```text
admin / admin
```

## API Usage

Health check:

```bash
curl http://localhost:8000/health
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 1000,
    "V1": -1.36,
    "V2": -0.07,
    "V3": 2.54,
    "V4": 1.38,
    "V5": -0.34,
    "V6": 0.46,
    "V7": 0.24,
    "V8": 0.10,
    "V9": 0.36,
    "V10": 0.09,
    "V11": -0.55,
    "V12": -0.62,
    "V13": -0.99,
    "V14": -0.31,
    "V15": 1.47,
    "V16": -0.47,
    "V17": 0.21,
    "V18": 0.03,
    "V19": 0.40,
    "V20": 0.25,
    "V21": -0.02,
    "V22": 0.28,
    "V23": -0.11,
    "V24": 0.07,
    "V25": 0.13,
    "V26": -0.19,
    "V27": 0.13,
    "V28": -0.02,
    "Amount": 149.62
  }'
```

Recent fraud alerts:

```bash
curl http://localhost:8000/alerts
```

## Load Testing

Run the k6 benchmark:

```bash
k6 run load-testing/script.js
```

For a deployed cloud URL:

```bash
BASE_URL=https://your-service-url k6 run load-testing/script.js
```

The script checks:

- P95 latency under 500 ms
- Error rate under 1%
- Valid prediction response body

## AWS Free Tier Deployment

This project uses Elastic Beanstalk because it is the easiest AWS path for a Dockerized API. Elastic Beanstalk does not add a separate platform charge, but the underlying EC2/S3/CloudWatch resources can use credits or free-tier capacity depending on your account. For the cheapest class-demo setup, create a single-instance environment and terminate it after collecting results.

Official AWS pricing pages:

- Elastic Beanstalk has no additional service charge; you pay for the AWS resources it creates.
- New AWS accounts can use Free Tier credits/free-plan benefits, but exact eligibility depends on account creation date and selected plan.

Package the deployment bundle:

```bash
./scripts/package_eb.sh
```

Deploy through the AWS Console:

1. Open Elastic Beanstalk.
2. Create application.
3. Choose **Web server environment**.
4. Platform: **Docker**.
5. Application code: upload `dist/fraud-detection-eb.zip`.
6. Presets: choose **Single instance** to avoid load balancer cost.
7. Instance type: choose the smallest free-tier/credit-friendly option available, such as `t3.micro` or `t2.micro`.
8. Create the environment and wait for the health status to turn green.

After deployment, test:

```bash
curl https://your-elastic-beanstalk-url/health
```

Run the cloud benchmark:

```bash
BASE_URL=https://your-elastic-beanstalk-url k6 run load-testing/script.js
```

Suggested evaluation:

1. Run k6 against the local Docker Compose API.
2. Deploy the API container to AWS Elastic Beanstalk.
3. Run the same k6 test against the cloud URL.
4. Compare P50/P95/P99 latency, throughput, and error rate.
5. Terminate the Elastic Beanstalk environment after the demo to avoid charges.

## Project Scope

Implemented:

- Containerized backend inference service
- Frontend dashboard for transaction scoring
- Local fraud alert endpoint
- Prometheus metrics endpoint
- Docker Compose multi-container deployment
- k6 quantitative load test
- AWS Elastic Beanstalk Docker deployment bundle

Future work:

- Kubernetes orchestration
- CI/CD deployment pipeline
- Persistent alert storage
- Model registry or ClearML experiment tracking
- Autoscaling comparison across multiple replicas
