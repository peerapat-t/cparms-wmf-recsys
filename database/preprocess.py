# This file converts the raw Amazon review dumps (JSON Lines, one review per line)
# into the flat interaction CSVs the experiments consume.

import argparse
import json
from pathlib import Path

import pandas as pd


DATABASE_DIR = Path(__file__).resolve().parent
JSON_DIR = DATABASE_DIR / "json"
CSV_DIR = DATABASE_DIR / "csv"

# Source dump -> output CSV, keyed by the short name used on the command line.
# The CSV names are the ones notebook_experiments.ipynb loads.
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

# Raw review field -> interaction column.
FIELD_MAP = {
    "reviewerID": "userid",
    "asin": "itemid",
    "unixReviewTime": "timestamp",
    "overall": "rating",
}

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"


# Inputs:
# - path: A raw Amazon dump; JSON Lines, one review object per line.
# Output: One row per review, holding only the four interaction fields.
# The review bodies dwarf the fields we keep, so lines are parsed one at a time and
# projected down immediately rather than loading every column into a DataFrame first.
def _read_reviews(path: Path) -> pd.DataFrame:
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


# Inputs:
# - df: Raw interaction rows, still carrying whatever types json.loads produced.
# Output: The same rows with final column types, and any row missing a field dropped.
def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["userid"] = df["userid"].astype("string")
    df["itemid"] = df["itemid"].astype("string")

    # unixReviewTime is seconds since the epoch, UTC. Naive datetimes are stored
    # (no tz suffix) because the splitter only ever compares timestamps to each other.
    epoch_seconds = pd.to_numeric(df["timestamp"], errors="coerce")
    df["timestamp"] = pd.to_datetime(epoch_seconds, unit="s", errors="coerce")

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("float64")

    return df.dropna(subset=["userid", "itemid", "timestamp", "rating"])


# Inputs:
# - df: Typed interaction rows, possibly repeating a (userid, itemid) pair.
# Output: At most one row per (userid, itemid).
# When a pair repeats, the earliest review is kept so a future review cannot move
# the user's first observed interaction to a later temporal split. Ties on timestamp
# are broken by the higher rating so the result does not depend on input row order.
def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(
        ["userid", "itemid", "timestamp", "rating"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )
    return ordered.drop_duplicates(subset=["userid", "itemid"], keep="first")


# Inputs:
# - name: Short dataset name, used for the console report.
# - source: Raw JSON dump to read.
# - target: CSV to write.
# Output: The written interaction table. The CSV is sorted by time, so it can be
# inspected as an event log; the splitter re-sorts it regardless.
# The source dumps are the published 5-core subsets, so no k-core filter is applied
# here: this stage only cleans what is given.
def preprocess(name: str, source: Path, target: Path) -> pd.DataFrame:
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


# Inputs: None; reads dataset selections and the output directory from the command line.
# Output: One CSV per selected dataset, written next to the raw dumps.
def main() -> None:
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
