"""Shared constants. Named here so live verification can change seed, k, or learning rate."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RANDOM_SEED = 42
MODEL_VERSION = "1.0.0"

FEATURE_COLUMNS = [
    "plot_area_ha",
    "rainfall_mm",
    "soil_ph",
    "seed_kg",
    "distance_km",
    "arrival_hour",
]
ID_COLUMN = "record_id"
REGRESSION_TARGET = "actual_yield_kg"
CLASSIFICATION_TARGET = "dispatch_attention"
EXPECTED_COLUMNS = [
    ID_COLUMN,
    *FEATURE_COLUMNS,
    REGRESSION_TARGET,
    CLASSIFICATION_TARGET,
]
NUMERIC_COLUMNS = FEATURE_COLUMNS + [REGRESSION_TARGET, CLASSIFICATION_TARGET]

TEST_SIZE = 0.2
LEARNING_RATE = 0.05
N_ITERATIONS = 3000
K_MIN = 2
K_MAX = 5
CLASSIFIER_NAME = "LogisticRegression"
