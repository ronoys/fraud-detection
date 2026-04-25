#!/bin/sh
set -eu
echo "Checking bundled model artifacts..."
ls -lh /var/app/staging/model/artifacts || true
