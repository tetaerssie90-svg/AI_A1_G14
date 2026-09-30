#!/usr/bin/env python3
"""Run the full HarvestLink decision pipeline and write assessor artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import MODEL_VERSION, RANDOM_SEED
from src.classification import run_classification
from src.clustering import run_clustering
from src.data import run_data_pipeline
from src.regression import run_regression
from src.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Musanze HarvestLink cooperative ML pipeline (SWE 3513 A1)."
    )
    parser.add_argument("--data", required=True, help="Path to AI_A1_GXX.csv")
    parser.add_argument("--output", required=True, help="Artifact output folder")
    parser.add_argument("--group", required=True, help="Group code, e.g. AI-G14")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.data)
    output_dir = ensure_dir(Path(args.output))
    models_dir = ensure_dir(Path(__file__).resolve().parent / "models")
    group_code = args.group.strip()

    print("=== HarvestLink pipeline ===")
    data_result = run_data_pipeline(csv_path, output_dir, group_code)
    df = data_result["frame"]
    run_regression(df, output_dir, models_dir, group_code)
    run_classification(df, output_dir, models_dir, group_code)
    run_clustering(df, output_dir, models_dir, group_code)

    write_json(
        models_dir / "meta.json",
        {
            "group_code": group_code,
            "model_version": MODEL_VERSION,
            "random_seed": RANDOM_SEED,
            "dataset_path": str(csv_path),
            "sha256_fingerprint": data_result["fingerprint"],
        },
    )
    print("Models saved in models/")
    print("Artifacts saved in", output_dir)
    print("Pipeline complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
