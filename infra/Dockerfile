FROM python:3.11-slim

# Create non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Install dependencies before copying source (better layer caching)
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY api/ .

# Ensure artifacts directory exists (model volume will be mounted here)
RUN mkdir -p model/artifacts

RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
