
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("features")


def engineer_features(input_path: str, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    # Create additional numerical features
    df["radius_perimeter_ratio"] = (
        df["mean radius"] /
        df["mean perimeter"].replace(0, 1)
    )

    df["area_perimeter_ratio"] = (
        df["mean area"] /
        df["mean perimeter"].replace(0, 1)
    )

    # Create radius category
    df["radius_category"] = pd.cut(
        df["mean radius"],
        bins=[0, 12, 18, float("inf")],
        labels=["small", "medium", "large"]
    ).astype(str)

    df.to_csv(output_path, index=False)

    logger.info(
        "Engineered features for %d rows -> %s",
        len(df),
        output_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/processed/cancer_clean.csv"
    )

    parser.add_argument(
        "--output",
        default="data/processed/cancer_features.csv"
    )

    args = parser.parse_args()

    engineer_features(args.input, args.output)
