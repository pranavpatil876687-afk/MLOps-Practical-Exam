
import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("preprocess")


def preprocess(input_path: str, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path).drop_duplicates()

    df.drop(
        columns=["collected_at"],
        inplace=True,
        errors="ignore"
    )

    for col in [c for c in df.columns if c != "diagnosis"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=["diagnosis"])

    df.to_csv(output_path, index=False)

    logger.info(
        "Preprocessed %d rows -> %s",
        len(df),
        output_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/raw/cancer_raw.csv"
    )

    parser.add_argument(
        "--output",
        default="data/processed/cancer_clean.csv"
    )

    args = parser.parse_args()

    preprocess(args.input, args.output)
