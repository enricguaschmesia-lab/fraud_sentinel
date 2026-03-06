from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from fraud_sentinel.data import RAW_FEATURE_COLUMNS


ENGINEERED_COLUMNS = [
    "log_amount",
    "amount_fractional",
    "hour_sin",
    "hour_cos",
    "day_index",
]


@dataclass
class FraudFeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds lightweight, row-local features that remain safe for batch inference."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FraudFeatureBuilder":
        _ = y
        self.feature_columns_ = self.get_feature_names_out()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._ensure_frame(X)
        transformed = frame.copy()
        transformed["log_amount"] = np.log1p(transformed["Amount"].clip(lower=0.0))
        transformed["amount_fractional"] = np.modf(transformed["Amount"])[0]

        hour_of_day = (transformed["Time"] % 86_400) / 3_600.0
        transformed["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24.0)
        transformed["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24.0)
        transformed["day_index"] = np.floor_divide(transformed["Time"], 86_400).astype(float)
        return transformed

    def get_feature_names_out(self) -> list[str]:
        return [*RAW_FEATURE_COLUMNS, *ENGINEERED_COLUMNS]

    @staticmethod
    def _ensure_frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            frame = pd.DataFrame(X, columns=RAW_FEATURE_COLUMNS)
        else:
            frame = X.copy()

        missing_columns = [column for column in RAW_FEATURE_COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Missing required feature columns: {missing_columns}")

        return frame[RAW_FEATURE_COLUMNS]

