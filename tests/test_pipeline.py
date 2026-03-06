from __future__ import annotations

from pathlib import Path

from fraud_sentinel.config import TrainingConfig
from fraud_sentinel.pipeline import run_training


def test_training_pipeline_generates_artifacts(
    synthetic_dataset_path: Path,
    workspace_tmp_dir: Path,
):
    artifact_dir = workspace_tmp_dir / "artifacts"
    figures_dir = artifact_dir / "figures"

    config = TrainingConfig(
        raw_data_path=synthetic_dataset_path,
        processed_dir=workspace_tmp_dir / "processed",
        artifact_dir=artifact_dir,
        figures_dir=figures_dir,
        model_bundle_path=artifact_dir / "bundle.joblib",
        metrics_path=artifact_dir / "metrics.json",
        leaderboard_path=artifact_dir / "leaderboard.csv",
        predictions_path=artifact_dir / "predictions.csv",
        drift_report_path=artifact_dir / "drift.csv",
        feature_importance_path=artifact_dir / "importance.csv",
        random_forest_estimators=60,
        balanced_random_forest_estimators=60,
        isolation_forest_estimators=60,
    )

    metrics = run_training(config)

    assert config.model_bundle_path.exists()
    assert config.metrics_path.exists()
    assert config.leaderboard_path.exists()
    assert config.predictions_path.exists()
    assert metrics["champion"]["model_name"] in {"logistic_regression", "random_forest", "balanced_random_forest", "isolation_forest"}
