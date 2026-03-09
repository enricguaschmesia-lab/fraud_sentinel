from __future__ import annotations

from pathlib import Path

from fraud_sentinel.config import DiagnosticsConfig, TrainingConfig, TuningConfig
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
        anomaly_leaderboard_path=artifact_dir / "anomaly_leaderboard.csv",
        tuning_results_path=artifact_dir / "tuning_results.csv",
        strategy_comparison_path=artifact_dir / "strategy_comparison.csv",
        class_distribution_path=artifact_dir / "class_distribution.csv",
        correlation_report_path=artifact_dir / "correlation.csv",
        error_analysis_path=artifact_dir / "error_analysis.csv",
        random_forest_estimators=60,
        balanced_random_forest_estimators=60,
        isolation_forest_estimators=60,
        smote_random_forest_estimators=50,
        diagnostics=DiagnosticsConfig(enabled=True, learning_curve_enabled=False),
        tuning=TuningConfig(enabled=False),
    )

    metrics = run_training(config)

    assert config.model_bundle_path.exists()
    assert config.metrics_path.exists()
    assert config.leaderboard_path.exists()
    assert config.predictions_path.exists()
    assert config.strategy_comparison_path.exists()
    assert config.class_distribution_path.exists()
    assert config.correlation_report_path.exists()
    assert config.error_analysis_path.exists()
    assert metrics["champion"]["model_name"] in {
        "weighted_logistic_regression",
        "random_forest",
        "balanced_random_forest",
        "smote_logistic_regression",
        "smote_random_forest",
        "nearmiss_logistic_regression",
    }
    assert "resampling_strategy_comparison" in metrics
    assert "model_card" in metrics
    assert "diagnostics" in metrics
