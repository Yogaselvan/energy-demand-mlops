"""
monitor.py
----------
Data drift monitoring using Evidently AI.

This script compares a reference (baseline) dataset against a current
(production) dataset and generates a comprehensive Data Drift HTML report.

Workflow:
  1. Load baseline dataset (data/baseline.csv) — created during training
  2. Load or generate a "current" dataset (data/current.csv)
  3. Run Evidently DataDriftPreset analysis
  4. Save HTML report to reports/data_drift_report.html
  5. Print a summary of detected drift to the console

The "current" dataset simulates production data by introducing controlled
distribution shifts across features.

Usage:
    python src/monitor.py [--current-data PATH]
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASELINE_PATH = os.path.join("data", "baseline.csv")
CURRENT_PATH = os.path.join("data", "current.csv")
REPORTS_DIR = "reports"
REPORT_PATH = os.path.join(REPORTS_DIR, "data_drift_report.html")

FEATURE_COLS = ["temperature", "humidity", "day_of_week"]
TARGET_COL = "energy_demand"

# Number of synthetic "current" records to generate if current CSV is absent
NUM_CURRENT_ROWS = 200
RANDOM_SEED = 99


def load_baseline(baseline_path: str) -> pd.DataFrame:
    """
    Load the reference (training-time) dataset.

    Args:
        baseline_path: Path to baseline CSV file.

    Returns:
        Baseline DataFrame.

    Raises:
        FileNotFoundError: If baseline CSV does not exist.
    """
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"Baseline dataset not found at '{baseline_path}'. "
            "Run `python src/train.py` first to generate it."
        )
    df = pd.read_csv(baseline_path)
    logger.info("Baseline loaded — shape: %s.", df.shape)
    return df


def generate_current_data(
    baseline_df: pd.DataFrame, n_rows: int = NUM_CURRENT_ROWS
) -> pd.DataFrame:
    """
    Generate a synthetic "current" dataset that mimics production drift.

    Introduces intentional distribution shifts:
      - Temperature: shifted up by +5°C to simulate a hotter period
      - Humidity:    increased variance to simulate irregular weather
      - day_of_week: random (independent of original)
      - energy_demand: derived from shifted features + noise

    Args:
        baseline_df: The reference dataset to base distributions on.
        n_rows:      Number of rows in the synthetic current dataset.

    Returns:
        Current DataFrame with the same columns as baseline.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # Shifted temperature distribution (+5°C drift)
    temp_mean = float(baseline_df["temperature"].mean()) + 5
    temp_std = float(baseline_df["temperature"].std()) * 1.2
    temperature = np.round(rng.normal(loc=temp_mean, scale=temp_std, size=n_rows), 2)

    # Higher-variance humidity
    humidity_mean = float(baseline_df["humidity"].mean())
    humidity = np.round(
        np.clip(rng.normal(loc=humidity_mean, scale=20.0, size=n_rows), 30, 95), 2
    )

    # Uniformly distributed days of week
    day_of_week = rng.integers(low=0, high=7, size=n_rows)

    # Energy demand derived from shifted features
    base = 500.0
    temp_effect = 8 * (temperature - 18) ** 2
    weekend_discount = (day_of_week >= 5).astype(float) * 50
    noise = rng.normal(loc=0, scale=25, size=n_rows)
    energy_demand = np.round(
        np.clip(base + temp_effect - weekend_discount + noise, 100, None), 2
    )

    current_df = pd.DataFrame(
        {
            "temperature": temperature,
            "humidity": humidity,
            "day_of_week": day_of_week,
            "energy_demand": energy_demand,
        }
    )

    # Persist for future runs / debugging
    os.makedirs("data", exist_ok=True)
    current_df.to_csv(CURRENT_PATH, index=False)
    logger.info(
        "Synthetic current dataset generated and saved to '%s' (%d rows).",
        CURRENT_PATH,
        n_rows,
    )
    return current_df


def load_current_data(current_path: str, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load the current (production) dataset, or auto-generate if absent.

    Args:
        current_path: Path to the current CSV file.
        baseline_df:  Baseline DataFrame used as a fallback reference for generation.

    Returns:
        Current DataFrame.
    """
    if os.path.exists(current_path):
        df = pd.read_csv(current_path)
        logger.info("Current dataset loaded from '%s' — shape: %s.", current_path, df.shape)
        return df
    else:
        logger.warning(
            "Current dataset not found at '%s'. Generating synthetic drift data...",
            current_path,
        )
        return generate_current_data(baseline_df)


def build_drift_report(
    baseline_df: pd.DataFrame, current_df: pd.DataFrame
) -> Report:
    """
    Build an Evidently Data Drift report.

    Args:
        baseline_df: Reference dataset (training distribution).
        current_df:  Current dataset (production distribution).

    Returns:
        Fitted Evidently Report object.
    """
    column_mapping = ColumnMapping(
        target=TARGET_COL,
        numerical_features=FEATURE_COLS,
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=baseline_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )
    return report


def save_report(report: Report, output_path: str) -> None:
    """
    Persist the Evidently report as an HTML file.

    Args:
        report:      Fitted Evidently Report.
        output_path: Destination file path for the HTML report.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.save_html(output_path)
    logger.info("Drift report saved to '%s'.", output_path)


def print_drift_summary(report: Report) -> None:
    """
    Extract and print a human-readable drift summary to the console.

    Args:
        report: Fitted Evidently Report.
    """
    report_dict = report.as_dict()
    metrics = report_dict.get("metrics", [])

    logger.info("=" * 60)
    logger.info("DATA DRIFT SUMMARY")
    logger.info("=" * 60)

    drift_detected = False
    for metric in metrics:
        metric_id = metric.get("metric", "")
        result = metric.get("result", {})

        # DatasetDriftMetric contains overall drift flag
        if "DatasetDriftMetric" in metric_id:
            dataset_drift = result.get("dataset_drift", False)
            drifted_cols = result.get("number_of_drifted_columns", 0)
            total_cols = result.get("number_of_columns", 0)
            drift_share = result.get("share_of_drifted_columns", 0.0)
            drift_detected = dataset_drift
            logger.info(
                "Dataset Drift Detected: %s | Drifted Columns: %d / %d (%.0f%%)",
                "✅ YES" if dataset_drift else "❌ NO",
                drifted_cols,
                total_cols,
                drift_share * 100,
            )

        # ColumnDriftMetric contains per-column drift info
        if "ColumnDriftMetric" in metric_id:
            col_name = result.get("column_name", "unknown")
            col_drift = result.get("drift_detected", False)
            stat_test = result.get("stattest_name", "N/A")
            p_value = result.get("drift_score", float("nan"))
            logger.info(
                "  Column %-15s | Drift: %-3s | Test: %-20s | Score: %.4f",
                col_name,
                "YES" if col_drift else "NO",
                stat_test,
                p_value,
            )

    logger.info("=" * 60)
    if drift_detected:
        logger.warning(
            "⚠️  Significant data drift detected. "
            "Consider retraining the model with fresh data."
        )
    else:
        logger.info("✅ No significant dataset-level drift detected.")


def run_monitoring(current_data_path: str = CURRENT_PATH) -> None:
    """
    Full monitoring pipeline: load → compare → report → summarize.

    Args:
        current_data_path: Path to the current (production) dataset CSV.
    """
    logger.info("Starting drift monitoring pipeline...")

    # ── Load Data ─────────────────────────────────────────────────────────────
    baseline_df = load_baseline(BASELINE_PATH)
    current_df = load_current_data(current_data_path, baseline_df)

    # ── Build & Save Report ───────────────────────────────────────────────────
    logger.info("Running Evidently Data Drift analysis...")
    report = build_drift_report(baseline_df, current_df)
    save_report(report, REPORT_PATH)

    # ── Console Summary ───────────────────────────────────────────────────────
    print_drift_summary(report)

    logger.info(
        "Monitoring complete. Open '%s' in a browser to view the full report.",
        REPORT_PATH,
    )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Evidently AI data drift monitoring."
    )
    parser.add_argument(
        "--current-data",
        type=str,
        default=CURRENT_PATH,
        help=f"Path to the current dataset CSV (default: '{CURRENT_PATH}').",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_monitoring(current_data_path=args.current_data)
