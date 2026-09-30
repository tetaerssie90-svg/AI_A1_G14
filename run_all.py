import pandas as pd
import numpy as np
import hashlib
import json
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--group", required=True, help="Group code")
    args = parser.parse_args()

    
    df = pd.read_csv(args.data)

    
    expected_columns = [
        "record_id", "plot_area_ha", "rainfall_mm", "soil_ph",
        "seed_kg", "distance_km", "arrival_hour",
        "actual_yield_kg", "dispatch_attention"
    ]

    
    if list(df.columns) != expected_columns:
        raise ValueError("Dataset schema does not match expected columns!")

    
    missing = df.isnull().sum().to_dict()
    duplicates = df.duplicated().sum()

    
    features = df.drop(columns=["record_id", "actual_yield_kg", "dispatch_attention"])
    feature_matrix = features.to_numpy()

    
    with open(args.data, "rb") as f:
        fingerprint = hashlib.sha256(f.read()).hexdigest()

    
    stats = df.describe().to_dict()

    
    report = {
        "row_count": len(df),
        "feature_count": features.shape[1],
        "missing_values": missing,
        "duplicates": int(duplicates),
        "descriptive_statistics": stats,
        "group_code": args.group,
        "sha256_fingerprint": fingerprint
    }

    
    with open(f"{args.output}/data_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Data pipeline complete. Report saved to data_report.json")

if __name__ == "__main__":
    main()
