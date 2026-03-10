# ============================================================
# Dockerfile — Energy Grid Demand Prediction API
# Production-ready FastAPI container
# ============================================================

# ── Stage: Base ─────────────────────────────────────────────
FROM python:3.11-slim AS base

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install OS-level dependencies (minimal footprint)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root application user for security
RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --no-create-home appuser

# ── Stage: Dependencies ──────────────────────────────────────
FROM base AS dependencies

WORKDIR /app

# Copy only the requirements file first to leverage Docker layer caching.
# Dependencies are re-installed only when requirements.txt changes.
COPY requirements.txt .

RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage: Application ───────────────────────────────────────
FROM dependencies AS application

WORKDIR /app

# Copy application source code
COPY src/ ./src/

# Copy MLflow model artifacts (populated by running train.py locally first)
# In CI/CD this would be fetched from a remote tracking server instead.
COPY mlruns/ ./mlruns/
COPY models/ ./models/

# Copy baseline data for drift monitoring reference
COPY data/ ./data/

# Create the reports directory (for drift reports at runtime)
RUN mkdir -p reports && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose the application port
EXPOSE 8000

# Health check — polls the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ───────────────────────────────────────────────
# Run via uvicorn with production settings (single worker per container;
# scale horizontally with multiple container replicas)
CMD ["uvicorn", "src.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
