from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fraud_sentinel.features import FraudFeatureBuilder
from fraud_sentinel.models import score_model


def load_bundle(bundle_path: Path | str) -> dict[str, Any]:
    return joblib.load(bundle_path)


def score_transactions(
    transactions: pd.DataFrame,
    bundle: dict[str, Any],
    threshold_profile: str = "balanced_f2",
) -> pd.DataFrame:
    raw_frame = transactions.copy()
    raw_frame = raw_frame.drop(columns=["Class"], errors="ignore")

    feature_builder: FraudFeatureBuilder = bundle["feature_builder"]
    feature_frame = feature_builder.transform(raw_frame)
    scores = score_model(bundle["model"], feature_frame)

    thresholds = bundle["thresholds"]
    if threshold_profile not in thresholds:
        raise ValueError(
            f"Unknown threshold profile '{threshold_profile}'. Available profiles: {sorted(thresholds)}"
        )

    threshold_value = thresholds[threshold_profile]["threshold"]
    predictions = scores >= threshold_value
    reason_codes = _reason_codes(
        feature_frame,
        feature_means=bundle["feature_stats"]["mean"],
        feature_stds=bundle["feature_stats"]["std"],
    )

    result = raw_frame.copy()
    result["risk_score"] = scores
    result["threshold_profile"] = threshold_profile
    result["threshold_used"] = threshold_value
    result["predicted_label"] = predictions.astype(int)
    result["predicted_label_name"] = np.where(predictions, "fraud", "legitimate")
    result["reason_codes"] = reason_codes
    return result


def _reason_codes(
    feature_frame: pd.DataFrame,
    feature_means: dict[str, float],
    feature_stds: dict[str, float],
    top_k: int = 3,
) -> list[str]:
    ordered_columns = list(feature_frame.columns)
    means = pd.Series(feature_means, index=ordered_columns, dtype=float)
    stds = pd.Series(feature_stds, index=ordered_columns, dtype=float).replace(0.0, 1.0)

    z_scores = ((feature_frame - means) / stds).abs()
    reasons: list[str] = []
    for _, row in z_scores.iterrows():
        top_features = row.nlargest(top_k)
        reasons.append(
            ", ".join(f"{feature}:{value:.1f}z" for feature, value in top_features.items())
        )
    return reasons
