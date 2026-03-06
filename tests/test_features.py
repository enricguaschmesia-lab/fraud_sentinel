from __future__ import annotations

from fraud_sentinel.features import ENGINEERED_COLUMNS, FraudFeatureBuilder


def test_feature_builder_adds_expected_columns(raw_feature_frame):
    builder = FraudFeatureBuilder()
    transformed = builder.fit_transform(raw_feature_frame)

    for column in ENGINEERED_COLUMNS:
        assert column in transformed.columns
    assert transformed.isna().sum().sum() == 0
    assert len(transformed.columns) == len(raw_feature_frame.columns) + len(ENGINEERED_COLUMNS)

