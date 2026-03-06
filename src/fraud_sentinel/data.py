from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fraud_sentinel.config import SplitConfig


RAW_FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET_COLUMN = "Class"
REQUIRED_COLUMNS = [*RAW_FEATURE_COLUMNS, TARGET_COLUMN]


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_dataset(csv_path: Path, sample_rows: int | None = None) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}.")

    frame = pd.read_csv(csv_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    frame = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    frame["Class"] = frame["Class"].astype(int)
    if sample_rows is not None:
        frame = frame.iloc[:sample_rows].copy()
    return frame


def split_dataset(frame: pd.DataFrame, split_config: SplitConfig) -> DatasetSplits:
    split_config.validate()

    total_rows = len(frame)
    train_end = int(total_rows * split_config.train_fraction)
    validation_end = train_end + int(total_rows * split_config.validation_fraction)

    if train_end <= 0 or validation_end <= train_end or validation_end >= total_rows:
        raise ValueError("Invalid split fractions for the dataset size.")

    return DatasetSplits(
        train=frame.iloc[:train_end].copy(),
        validation=frame.iloc[train_end:validation_end].copy(),
        test=frame.iloc[validation_end:].copy(),
    )

