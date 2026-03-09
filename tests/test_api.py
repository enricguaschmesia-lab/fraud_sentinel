from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_sentinel.api import create_app
from fraud_sentinel.config import DiagnosticsConfig, TrainingConfig, TuningConfig
from fraud_sentinel.pipeline import run_training


def test_prediction_api_returns_scores(
    synthetic_dataset_path: Path,
    workspace_tmp_dir: Path,
):
    artifact_dir = workspace_tmp_dir / "artifacts"
    config = TrainingConfig(
        raw_data_path=synthetic_dataset_path,
        processed_dir=workspace_tmp_dir / "processed",
        artifact_dir=artifact_dir,
        figures_dir=artifact_dir / "figures",
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
        random_forest_estimators=40,
        balanced_random_forest_estimators=40,
        isolation_forest_estimators=40,
        smote_random_forest_estimators=40,
        tuning=TuningConfig(enabled=False),
        diagnostics=DiagnosticsConfig(enabled=False, learning_curve_enabled=False),
    )
    run_training(config)

    frame = pd.read_csv(synthetic_dataset_path).head(3).drop(columns=["Class"])
    app = create_app(config.model_bundle_path)
    client = app.test_client()
    response = client.post(
        "/predict",
        json={
            "threshold_profile": "balanced_f2",
            "records": frame.to_dict(orient="records"),
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload["predictions"]) == 3
    assert "risk_score" in payload["predictions"][0]
