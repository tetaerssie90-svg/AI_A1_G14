#!/usr/bin/env python3
"""Score one collection record with the trained HarvestLink models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, MODEL_VERSION
from src.regression import add_bias, apply_standardizer
from src.utils import load_joblib

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
REQUIRED_FIELDS = FEATURE_COLUMNS


class RecordError(ValueError):
    """Raised when the input JSON is missing or malformed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict harvest weight, dispatch attention, and cluster."
    )
    parser.add_argument(
        "--record",
        required=True,
        help="JSON object with the six input fields from the assignment schema.",
    )
    return parser.parse_args()


def parse_record(raw: str) -> dict[str, float]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RecordError(f"Malformed JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise RecordError("Record must be a JSON object.")

    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise RecordError(f"Missing fields: {', '.join(missing)}")

    parsed: dict[str, float] = {}
    for field in REQUIRED_FIELDS:
        value = payload[field]
        if isinstance(value, bool) or value is None:
            raise RecordError(f"Field '{field}' must be a finite number.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise RecordError(f"Field '{field}' must be a finite number.") from exc
        if not np.isfinite(number):
            raise RecordError(f"Field '{field}' must be a finite number.")
        parsed[field] = number
    return parsed


def vector_from_record(record: dict[str, float]) -> np.ndarray:
    return np.array([[record[name] for name in FEATURE_COLUMNS]], dtype=float)


def load_meta() -> dict:
    path = MODELS / "meta.json"
    if not path.exists():
        raise FileNotFoundError("models/meta.json is missing. Run run_all.py first.")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def predict(record: dict[str, float]) -> dict:
    meta = load_meta()
    regression = load_joblib(MODELS / "regression.joblib")
    classifier_bundle = load_joblib(MODELS / "classifier.joblib")
    cluster_bundle = load_joblib(MODELS / "cluster.joblib")

    X = vector_from_record(record)
    X_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    X_reg = apply_standardizer(X, regression["mean"], regression["std"])
    harvest_kg = float(add_bias(X_reg) @ regression["theta"])

    pipeline = classifier_bundle["pipeline"]
    class_label = int(pipeline.predict(X_df)[0])
    probabilities = pipeline.predict_proba(X_df)[0]
    classes = list(pipeline.classes_)
    attention_index = classes.index(1) if 1 in classes else int(np.argmax(probabilities))
    attention_probability = float(probabilities[attention_index])

    X_cluster = cluster_bundle["scaler"].transform(X_df)
    cluster_label = int(cluster_bundle["model"].predict(X_cluster)[0])

    return {
        "regression_prediction_kg": harvest_kg,
        "classification_prediction": class_label,
        "classification_probability": attention_probability,
        "cluster_label": cluster_label,
        "group_code": meta.get("group_code", "UNKNOWN"),
        "model_version": meta.get("model_version", MODEL_VERSION),
    }


def main() -> int:
    args = parse_args()
    try:
        record = parse_record(args.record)
        result = predict(record)
    except RecordError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    except FileNotFoundError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
