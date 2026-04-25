#!/bin/sh
set -eu

mkdir -p dist
rm -f dist/fraud-detection-eb.zip

zip -r dist/fraud-detection-eb.zip \
  Dockerfile \
  api \
  model/artifacts/xgboost.joblib \
  model/artifacts/scaler.joblib \
  .ebextensions \
  .platform \
  -x '*/__pycache__/*' '*.pyc' '.DS_Store'

echo "Created dist/fraud-detection-eb.zip"
