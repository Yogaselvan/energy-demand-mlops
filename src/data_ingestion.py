"""
data_ingestion.py
-----------------
Generates mock historical weather and energy demand data and saves it as a CSV.

The synthetic dataset simulates 3 years of daily readings with realistic
seasonal patterns:
  - Temperature: sinusoidal seasonal variation + random noise
  - Humidity:    random variation bounded between 30% and 95%
  - Day of Week: derived from date (0 = Monday, 6 = Sunday)
  - Energy Demand: correlated with temperature (higher in summer/winter)
                   and day of week (lower on weekends).

Output:
    data/energy_demand.csv
"""

import os
import logging
import numpy as np
import pandas as pd

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
RANDOM_SEED = 42
NUM_DAYS = 365 * 3          # 3 years of daily data
START_DATE = "2021-01-01"
OUTPUT_PATH = os.path.join("data", "energy_demand.csv")


def generate_temperature(num_days: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate synthetic daily temperature values (°C).

    Uses a sinusoidal curve to model seasonal variation (peaks in summer,
    troughs in winter) with added Gaussian noise.

    Args:
        num_days: Number of daily records to generate.
        rng:      NumPy random Generator for reproducibility.

    Returns:
        Array of temperature values in °C.
    """
    days = np.arange(num_days)
    # Peak at day ~180 (mid-year summer), amplitude ±15°C around mean of 15°C
    seasonal = 15 + 15 * np.sin(2 * np.pi * (days - 80) / 365)
    noise = rng.normal(loc=0, scale=3.0, size=num_days)
    return np.round(seasonal + noise, 2)


def generate_humidity(num_days: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate synthetic daily relative humidity values (%).

    Clipped to physically realistic bounds [30, 95].

    Args:
        num_days: Number of daily records to generate.
        rng:      NumPy random Generator for reproducibility.

    Returns:
        Array of humidity values in %.
    """
    humidity = rng.uniform(low=30, high=95, size=num_days)
    return np.round(np.clip(humidity, 30, 95), 2)


def generate_energy_demand(
    temperature: np.ndarray,
    day_of_week: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate synthetic daily energy demand values (MWh).

    Demand is driven by:
      - Temperature: higher demand at extremes (U-shaped: hot summers, cold winters)
      - Weekday vs weekend: ~10% lower demand on weekends (days 5 & 6)
      - Gaussian noise for realistic variance

    Args:
        temperature:  Array of daily temperature values.
        day_of_week:  Array of day-of-week indices (0=Mon, 6=Sun).
        rng:          NumPy random Generator for reproducibility.

    Returns:
        Array of energy demand values in MWh.
    """
    base_demand = 500.0  # MWh baseline

    # U-shaped temperature effect: demand rises as temp moves away from 18°C
    temp_effect = 8 * (temperature - 18) ** 2

    # Weekend discount: ~10% reduction Saturday & Sunday
    weekend_mask = (day_of_week >= 5).astype(float)
    weekend_discount = weekend_mask * 50

    noise = rng.normal(loc=0, scale=20, size=len(temperature))

    demand = base_demand + temp_effect - weekend_discount + noise
    return np.round(np.clip(demand, 100, None), 2)  # Demand cannot be negative


def generate_dataset(num_days: int = NUM_DAYS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Orchestrate synthetic dataset generation.

    Args:
        num_days: Number of daily records to create.
        seed:     Random seed for reproducibility.

    Returns:
        DataFrame with columns: date, temperature, humidity,
                                day_of_week, energy_demand.
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=START_DATE, periods=num_days, freq="D")
    temperature = generate_temperature(num_days, rng)
    humidity = generate_humidity(num_days, rng)
    day_of_week = dates.day_of_week.to_numpy()  # 0=Monday, 6=Sunday
    energy_demand = generate_energy_demand(temperature, day_of_week, rng)

    df = pd.DataFrame(
        {
            "date": dates,
            "temperature": temperature,
            "humidity": humidity,
            "day_of_week": day_of_week,
            "energy_demand": energy_demand,
        }
    )
    return df


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Persist the generated dataset to a CSV file.

    Args:
        df:          DataFrame to save.
        output_path: Relative or absolute path to the output CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Dataset saved to '%s' (%d rows).", output_path, len(df))


def main() -> None:
    """Entry point: generate and persist the energy demand dataset."""
    logger.info("Starting data ingestion — generating %d days of data.", NUM_DAYS)
    df = generate_dataset()
    logger.info(
        "Dataset generated. Summary:\n%s",
        df.describe().to_string(),
    )
    save_dataset(df, OUTPUT_PATH)
    logger.info("Data ingestion complete.")


if __name__ == "__main__":
    main()
