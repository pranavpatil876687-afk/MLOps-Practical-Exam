
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("validate")


def validate_data(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    required_columns = [
        "diagnosis",
        "mean radius",
        "mean perimeter",
        "mean area",
        "radius_category"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if len(df) == 0:
        raise ValueError("Dataset is empty")

    if df["diagnosis"].isna().any():
        raise ValueError("Missing diagnosis values found")

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.isna().any().any():
        raise ValueError("Missing numeric values found")

    if not np.isfinite(numeric_df.to_numpy()).all():
        raise ValueError("Infinite numeric values found")

    report = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "diagnosis_classes": df["diagnosis"].unique().tolist(),
        "validation_status": "PASSED"
    }

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    logger.info("Data validation PASSED")
    logger.info("Validation report saved -> %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/processed/cancer_features.csv"
    )

    parser.add_argument(
        "--output",
        default="data/processed/validation_report.json"
    )

    args = parser.parse_args()

    validate_data(args.input, args.output)
