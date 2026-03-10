"""
train.py
--------
Trains a Random Forest Regressor on the energy demand dataset and tracks
the experiment with MLflow (parameters, metrics, and model artifact).

Workflow:
  1. Load data from data/energy_demand.csv
  2. Split into time-ordered train/test sets (80/20)
  3. Train a RandomForestRegressor
  4. Evaluate: compute RMSE and R²
  5. Log everything with MLflow (params, metrics, model)
  6. Register the model in the MLflow Model Registry
  7. Save baseline data for drift monitoring (data/baseline.csv)

Usage:
    python src/train.py
"""

import os
import logging
import warnings

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "energy_demand.csv")
BASELINE_PATH = os.path.join("data", "baseline.csv")
MLFLOW_EXPERIMENT = "energy-demand-forecasting"
MODEL_REGISTRY_NAME = "energy-demand-model"

# Model hyperparameters — adjust here to track different experiment runs
N_ESTIMATORS = 200
MAX_DEPTH = 15
MIN_SAMPLES_SPLIT = 5
RANDOM_STATE = 42
TRAIN_RATIO = 0.80

# Feature columns used for training
FEATURE_COLS = ["temperature", "humidity", "day_of_week"]
TARGET_COL = "energy_demand"


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the energy demand dataset from a CSV file.

    Args:
        data_path: Path to the CSV file.

    Returns:
        DataFrame with all columns as parsed from the CSV.

    Raises:
        FileNotFoundError: If the CSV does not exist at the given path.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            "Run `python src/data_ingestion.py` first."
        )
    df = pd.read_csv(data_path, parse_dates=["date"])
    logger.info("Loaded dataset from '%s' — shape: %s.", data_path, df.shape)
    return df


def split_data(
    df: pd.DataFrame, train_ratio: float = TRAIN_RATIO
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into time-ordered train and test sets.

    Time-based splitting avoids data leakage that would occur with
    random shuffling on time-series data.

    Args:
        df:          Full feature+target DataFrame.
        train_ratio: Fraction of data to use for training.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[FEATURE_COLS]
    X_test = test_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    logger.info(
        "Train/test split — Train: %d rows, Test: %d rows.",
        len(X_train),
        len(X_test),
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.

    Returns:
        Fitted RandomForestRegressor model.
    """
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        random_state=RANDOM_STATE,
        n_jobs=-1,  # Use all available CPU cores
    )
    logger.info("Training Random Forest with %d estimators...", N_ESTIMATORS)
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model


def evaluate_model(
    model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """
    Evaluate the trained model on the test set.

    Args:
        model:  Fitted RandomForestRegressor.
        X_test: Test feature matrix.
        y_test: True target values.

    Returns:
        Dictionary with 'rmse' and 'r2' metric values.
    """
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    logger.info("Evaluation — RMSE: %.4f | R²: %.4f", rmse, r2)
    return {"rmse": rmse, "r2": r2}


def save_baseline(df: pd.DataFrame, baseline_path: str) -> None:
    """
    Save the training portion of the dataset as the reference baseline
    for Evidently AI drift monitoring.

    Args:
        df:            Full DataFrame (pre-split).
        baseline_path: Output path for the baseline CSV.
    """
    split_idx = int(len(df) * TRAIN_RATIO)
    baseline_df = df.iloc[:split_idx][FEATURE_COLS + [TARGET_COL]]
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
    baseline_df.to_csv(baseline_path, index=False)
    logger.info("Baseline dataset saved to '%s' (%d rows).", baseline_path, len(baseline_df))


def run_training_pipeline() -> None:
    """
    Full training pipeline: load → split → train → evaluate → log to MLflow.

    The trained model is registered in the MLflow Model Registry under
    the name defined by MODEL_REGISTRY_NAME.
    """
    # ── Setup MLflow ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri("mlruns")  # Local tracking directory
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # ── Load & Split Data ────────────────────────────────────────────────────
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)

    # ── MLflow Run ────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="random_forest_v1") as run:
        run_id = run.info.run_id
        logger.info("MLflow run started — Run ID: %s", run_id)

        # Log hyperparameters
        params = {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_split": MIN_SAMPLES_SPLIT,
            "random_state": RANDOM_STATE,
            "train_ratio": TRAIN_RATIO,
            "features": str(FEATURE_COLS),
        }
        mlflow.log_params(params)

        # Train
        model = train_model(X_train, y_train)

        # Evaluate & log metrics
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Infer model signature from training data
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model artifact and register it in the Model Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_REGISTRY_NAME,
            input_example=X_train.iloc[:3],
        )

        # Save a portable copy directly to the models/ folder for Docker.
        # This avoids MLflow's absolute path registry issues when moving 
        # local models into containers.
        import shutil
        local_model_path = os.path.join("models", MODEL_REGISTRY_NAME)
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)
        mlflow.sklearn.save_model(
            sk_model=model,
            path=local_model_path,
            signature=signature,
            input_example=X_train.iloc[:3],
        )

        logger.info(
            "Model registered as '%s'. Run ID: %s", MODEL_REGISTRY_NAME, run_id
        )
        logger.info(
            "View results in MLflow UI: mlflow ui --port 5000 (http://localhost:5000)"
        )

    # ── Save Baseline for Drift Monitoring ───────────────────────────────────
    save_baseline(df, BASELINE_PATH)
    logger.info("Training pipeline finished successfully.")


if __name__ == "__main__":
    run_training_pipeline()
