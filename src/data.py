"""Load, validate, clean, and vectorize the lecturer-issued cooperative CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLASSIFICATION_TARGET,
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    REGRESSION_TARGET,
)
from src.utils import sha256_file, to_jsonable, write_json


class SchemaError(ValueError):
    """Raised when the CSV does not match the published assignment schema."""


def _validate_columns(df: pd.DataFrame) -> None:
    actual = list(df.columns)
    missing = [col for col in EXPECTED_COLUMNS if col not in actual]
    extra = [col for col in actual if col not in EXPECTED_COLUMNS]
    if missing or extra:
        raise SchemaError(
            "Dataset schema does not match the published assignment columns. "
            f"Missing: {missing or 'none'}. Unexpected: {extra or 'none'}."
        )


def _validate_types(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned[ID_COLUMN] = cleaned[ID_COLUMN].astype(str).str.strip()
    if cleaned[ID_COLUMN].eq("").any() or cleaned[ID_COLUMN].eq("nan").any():
        raise SchemaError("record_id must be a non-empty identifier on every row.")

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        invalid = cleaned[column].isna() & ~df[column].isna()
        if invalid.any():
            bad_ids = cleaned.loc[invalid, ID_COLUMN].tolist()
            raise SchemaError(
                f"Column '{column}' has non-numeric values in records {bad_ids}."
            )

    illegal_attention = ~cleaned[CLASSIFICATION_TARGET].dropna().isin([0, 1])
    if illegal_attention.any():
        bad_ids = cleaned.loc[illegal_attention, ID_COLUMN].tolist()
        raise SchemaError(
            f"dispatch_attention must be 0 or 1. Invalid records: {bad_ids}."
        )
    return cleaned


def load_and_validate(csv_path: Path) -> tuple[pd.DataFrame, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    fingerprint = sha256_file(csv_path)
    df = pd.read_csv(csv_path)
    _validate_columns(df)
    df = df[EXPECTED_COLUMNS]
    df = _validate_types(df)
    return df, fingerprint


def vectorize(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Return the NumPy feature matrix. Identifiers and targets are excluded."""
    features = df[FEATURE_COLUMNS].copy()
    matrix = features.to_numpy(dtype=float)
    return matrix, features


def build_data_report(df: pd.DataFrame, fingerprint: str, group_code: str) -> dict:
    _, features = vectorize(df)
    missing_values = df.isnull().sum().astype(int).to_dict()
    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df[ID_COLUMN].duplicated().sum())
    stats = df[NUMERIC_COLUMNS].describe().to_dict()
    return {
        "row_count": int(len(df)),
        "feature_count": int(features.shape[1]),
        "feature_names": FEATURE_COLUMNS,
        "identifier_column": ID_COLUMN,
        "regression_target": REGRESSION_TARGET,
        "classification_target": CLASSIFICATION_TARGET,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "duplicate_record_ids": duplicate_ids,
        "descriptive_statistics": to_jsonable(stats),
        "group_code": group_code,
        "sha256_fingerprint": fingerprint,
    }


def run_data_pipeline(csv_path: Path, output_dir: Path, group_code: str) -> dict:
    df, fingerprint = load_and_validate(csv_path)
    matrix, _ = vectorize(df)
    report = build_data_report(df, fingerprint, group_code)
    report["feature_matrix_shape"] = [int(matrix.shape[0]), int(matrix.shape[1])]
    write_json(output_dir / "data_report.json", report)
    print(f"Group code: {group_code}")
    print(f"Dataset SHA-256: {fingerprint}")
    print(
        f"Rows: {report['row_count']} | Features: {report['feature_count']} | "
        f"Missing cells: {int(sum(report['missing_values'].values()))}"
    )
    return {"frame": df, "fingerprint": fingerprint, "report": report, "X": matrix}
