
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from sklearn.datasets import load_breast_cancer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("collect")


def collect_data(output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    data = load_breast_cancer(as_frame=True)

    df = data.frame.rename(columns={"target": "diagnosis"})

    df["diagnosis"] = df["diagnosis"].map(
        dict(enumerate(data.target_names))
    )

    df["collected_at"] = datetime.now(timezone.utc).isoformat()

    df.to_csv(output_path, index=False)

    logger.info(
        "Collected %d rows -> %s",
        len(df),
        output_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="data/raw/cancer_raw.csv"
    )

    collect_data(parser.parse_args().output)
