<div align="center">
  <h1>⚡ Energy Demand MLOps Pipeline</h1>
  <p>A production-grade, end-to-end Machine Learning pipeline to predict daily energy grid demand based on weather data.</p>
</div>

---

## 🎯 Project Overview

This project demonstrates a complete MLOps workflow. It ingests synthetic weather and energy demand data, trains a Random Forest model with robust experiment tracking, serves predictions via a FastAPI backend, and monitors for data drift in production using Evidently AI. The entire prediction service is Dockerized for scalability and easy deployment.

### 🛠 Tech Stack
- **Modeling & Data processing:** Python, Scikit-learn, Pandas, Numpy
- **Experiment Tracking & Model Registry:** MLflow
- **API Serving:** FastAPI, Uvicorn, Pydantic
- **Data Drift Monitoring:** Evidently AI
- **Containerization:** Docker

---

## 📂 Repository Structure

```text
energy-demand-mlops/
├── data/                        # Raw & processed data CSVs (generated at runtime)
├── models/                      # Portable model artifacts for Docker
├── mlruns/                      # MLflow tracking directory (auto-created)
├── reports/                     # Drift monitoring HTML reports
├── src/
│   ├── data_ingestion.py        # Mock data generator (3 years of data)
│   ├── train.py                 # RF Training + MLflow tracking
│   ├── app.py                   # FastAPI prediction server
│   └── monitor.py               # Evidently AI data drift monitoring
├── requirements.txt             # Pinned Python dependencies
├── Dockerfile                   # Production Docker image (multi-stage)
└── .dockerignore                # Exclude unneeded files from Docker context
```

---

## 🚀 Getting Started (Local Development)

### 1. Setup Virtual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Data
Generates 1,095 days (3 years) of synthetic weather + energy demand data with realistic seasonal patterns.
```bash
python src/data_ingestion.py
```

### 3. Train the Model
Trains a Random Forest Regressor. Logs hyperparameters, RMSE, and R² to MLflow, and saves the model to the local `models/` directory for portable deployment.
```bash
python src/train.py
```
*(Optional)* View the MLflow UI: `mlflow ui --port 5000`

### 4. Start the API Server
Starts the FastAPI backend. The model is loaded into memory on startup (lifespan manager) to ensure low-latency predictions.
```bash
uvicorn src.app:app --reload --port 8000
```
> View the interactive Swagger API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Run Data Drift Monitoring
Compares the reference training data against current production data to detect feature drift using statistical tests.
```bash
python src/monitor.py
```
> Open `reports/data_drift_report.html` in your browser to view the interactive Evidently AI report.

---

## 🐳 Docker Deployment

The prediction API is fully containerized using a multi-stage, non-root Dockerfile for production readiness.

**1. Build the image**
```bash
docker build -t energy-demand-api .
```

**2. Run the container**
```bash
docker run -p 8000:8000 energy-demand-api
```

**3. Test the Endpoint**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 25.0,
    "humidity": 60.0,
    "day_of_week": 1
  }'
```
**Expected Response:**
```json
{
  "predicted_demand_mwh": 882.27,
  "model_name": "energy-demand-model",
  "model_version": "portable-local"
}
```

---

## 🧠 Key Design Decisions
- **Time-based train/test split:** Prevents data leakage in time-series data.
- **FastAPI Lifespan Loading:** The ML model is loaded exactly once during application startup, preventing per-request I/O bottlenecks.
- **Non-root Docker User:** Adheres to security best practices.
- **Portable Model Directory (`models/`):** Circumvents issues where the local MLflow registry stores absolute paths to artifacts which break when mounted inside a Docker container.
