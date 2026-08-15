"""Convert raw Amazon review dumps into clean interaction datasets."""

import argparse
import json
from pathlib import Path

import pandas as pd


DATABASE_DIR = Path(__file__).resolve().parent
JSON_DIR = DATABASE_DIR / "json"
CSV_DIR = DATABASE_DIR / "csv"


DATASETS = {
    "beauty": ("Luxury_Beauty_5.json", "dataset_amazon_lux_beauty_5_core.csv"),
    "industry": (
        "Industrial_and_Scientific_5.json",
        "dataset_amazon_industry_5_core.csv",
    ),
    "pantry": ("Prime_Pantry_5.json", "dataset_amazon_pantry_5_core.csv"),
    "music": ("Digital_Music_5.json", "dataset_amazon_music_5_core.csv"),
    "instruments": (
        "Musical_Instruments_5.json",
        "dataset_amazon_instruments_5_core.csv",
    ),
}


FIELD_MAP = {
    "reviewerID": "userid",
    "asin": "itemid",
    "unixReviewTime": "timestamp",
    "overall": "rating",
}

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _read_reviews(path: Path) -> pd.DataFrame:
    """Load the selected review fields from a JSON Lines file."""
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            review = json.loads(line)
            records.append(
                {
                    column: review.get(field)
                    for field, column in FIELD_MAP.items()
                }
            )

    if not records:
        raise ValueError(f"{path.name} contains no reviews")
    return pd.DataFrame.from_records(records, columns=list(FIELD_MAP.values()))


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce interaction columns and discard rows with invalid values."""
    df = df.copy()
    df["userid"] = df["userid"].astype("string")
    df["itemid"] = df["itemid"].astype("string")


    epoch_seconds = pd.to_numeric(df["timestamp"], errors="coerce")
    df["timestamp"] = pd.to_datetime(epoch_seconds, unit="s", errors="coerce")

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("float64")

    return df.dropna(subset=["userid", "itemid", "timestamp", "rating"])


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the earliest highest-rated record for each user-item pair."""
    ordered = df.sort_values(
        ["userid", "itemid", "timestamp", "rating"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )
    return ordered.drop_duplicates(subset=["userid", "itemid"], keep="first")


def preprocess(name: str, source: Path, target: Path) -> pd.DataFrame:
    """Clean one review dataset and write the resulting interaction CSV."""
    raw = _read_reviews(source)
    typed = _coerce_types(raw)
    deduped = _drop_duplicates(typed)

    final = deduped.sort_values(["timestamp", "userid", "itemid"], kind="mergesort")
    final = final.reset_index(drop=True)
    final.to_csv(target, index=False, date_format=TIMESTAMP_FORMAT)

    print(
        f"{name:12s} {len(raw):>8,} reviews -> {len(final):>8,} interactions"
        f" | dropped {len(raw) - len(typed):>5,} invalid,"
        f" {len(typed) - len(deduped):>6,} duplicate"
        f" | {final['userid'].nunique():>7,} users,"
        f" {final['itemid'].nunique():>6,} items"
    )
    return final


def main() -> None:
    """Parse command-line options and preprocess the requested datasets."""
    parser = argparse.ArgumentParser(
        description="Convert raw Amazon review dumps into interaction CSVs.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        action="append",
        help="Dataset to process; repeatable. Defaults to all of them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CSV_DIR,
        help="Directory to write the CSVs into. Defaults to database/csv/.",
    )
    args = parser.parse_args()

    names = args.dataset or sorted(DATASETS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        source_name, target_name = DATASETS[name]
        source = JSON_DIR / source_name
        if not source.exists():
            raise FileNotFoundError(f"missing raw dump: {source}")
        preprocess(name, source, args.output_dir / target_name)


if __name__ == "__main__":
    main()
