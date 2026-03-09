from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_sentinel.config import DiagnosticsConfig, TrainingConfig, TuningConfig
from fraud_sentinel.data import load_dataset
from fraud_sentinel.features import FraudFeatureBuilder
from fraud_sentinel.models import build_candidate_models
from fraud_sentinel.tuning import tune_estimator


def test_candidate_factory_includes_resampling_strategies():
    specs = build_candidate_models(TrainingConfig())
    strategy_names = {spec.name for spec in specs}

    assert "smote_logistic_regression" in strategy_names
    assert "smote_random_forest" in strategy_names
    assert "nearmiss_logistic_regression" in strategy_names


def test_tuning_returns_deterministic_summary(synthetic_dataset_path: Path, workspace_tmp_dir: Path):
    config = TrainingConfig(
        raw_data_path=synthetic_dataset_path,
        processed_dir=workspace_tmp_dir / "processed",
        artifact_dir=workspace_tmp_dir / "artifacts",
        figures_dir=workspace_tmp_dir / "artifacts" / "figures",
        tuning=TuningConfig(enabled=True, cv_folds=2, search_iterations=2, max_rows_for_search=800),
        diagnostics=DiagnosticsConfig(enabled=False, learning_curve_enabled=False),
    )
    frame = load_dataset(synthetic_dataset_path)
    feature_builder = FraudFeatureBuilder()
    X = feature_builder.fit_transform(frame.drop(columns=["Class"]))
    y = frame["Class"]

    spec = next(spec for spec in build_candidate_models(config) if spec.name == "weighted_logistic_regression")
    estimator = spec.builder(config)
    best_estimator, outcome, tuning_frame = tune_estimator(spec, estimator, X, y, config)

    assert outcome.tuned is True
    assert outcome.best_params
    assert not tuning_frame.empty
    assert best_estimator is not None
