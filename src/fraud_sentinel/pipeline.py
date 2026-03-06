from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fraud_sentinel.config import TrainingConfig
from fraud_sentinel.data import RAW_FEATURE_COLUMNS, TARGET_COLUMN, load_dataset, split_dataset
from fraud_sentinel.evaluation import (
    compute_threshold_profiles,
    drift_report,
    evaluate_fixed_thresholds,
    leaderboard_row,
)
from fraud_sentinel.features import FraudFeatureBuilder
from fraud_sentinel.inference import score_transactions
from fraud_sentinel.models import (
    build_candidate_models,
    extract_feature_importance,
    fit_model,
    score_model,
)
from fraud_sentinel.reporting import (
    plot_drift_report,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_threshold_tradeoffs,
    write_json,
    write_table,
)


def run_training(config: TrainingConfig | None = None) -> dict[str, Any]:
    config = config or TrainingConfig()
    config.ensure_directories()

    raw_frame = load_dataset(config.raw_data_path, sample_rows=config.sample_rows)
    splits = split_dataset(raw_frame, config.split)

    feature_builder = FraudFeatureBuilder()
    train_features = feature_builder.fit_transform(splits.train[RAW_FEATURE_COLUMNS])
    validation_features = feature_builder.transform(splits.validation[RAW_FEATURE_COLUMNS])
    test_features = feature_builder.transform(splits.test[RAW_FEATURE_COLUMNS])
    feature_names = train_features.columns.tolist()

    y_train = splits.train[TARGET_COLUMN]
    y_validation = splits.validation[TARGET_COLUMN].to_numpy()
    y_test = splits.test[TARGET_COLUMN].to_numpy()
    validation_amounts = splits.validation["Amount"].to_numpy()
    test_amounts = splits.test["Amount"].to_numpy()

    candidate_results: dict[str, dict[str, Any]] = {}
    leaderboard_rows: list[dict[str, Any]] = []

    for spec in build_candidate_models(config):
        estimator = fit_model(spec, train_features, y_train)
        validation_scores = score_model(estimator, validation_features)
        validation_profiles, threshold_frame = compute_threshold_profiles(
            y_true=y_validation,
            scores=validation_scores,
            amounts=validation_amounts,
            threshold_config=config.thresholds,
            business_config=config.business,
        )
        test_scores = score_model(estimator, test_features)
        test_profiles = evaluate_fixed_thresholds(
            y_true=y_test,
            scores=test_scores,
            amounts=test_amounts,
            thresholds=validation_profiles,
            business_config=config.business,
        )

        leaderboard_rows.append(
            leaderboard_row(
                model_name=spec.name,
                description=spec.description,
                validation_profiles=validation_profiles,
                test_profiles=test_profiles,
            )
        )
        candidate_results[spec.name] = {
            "description": spec.description,
            "estimator": estimator,
            "validation_profiles": validation_profiles,
            "test_profiles": test_profiles,
            "validation_threshold_curve": threshold_frame,
            "validation_scores": validation_scores,
            "test_scores": test_scores,
        }

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["selection_score", "validation_average_precision"],
        ascending=False,
        kind="stable",
    )
    write_table(leaderboard, config.leaderboard_path)

    champion_name = str(leaderboard.iloc[0]["model_name"])
    champion = candidate_results[champion_name]
    importance_frame = extract_feature_importance(champion["estimator"], feature_names)
    drift_frame = drift_report(train_features, test_features)
    write_table(importance_frame, config.feature_importance_path)
    write_table(drift_frame, config.drift_report_path)

    test_prediction_frame = score_transactions(
        splits.test[RAW_FEATURE_COLUMNS],
        bundle={
            "model": champion["estimator"],
            "feature_builder": feature_builder,
            "thresholds": {
                name: result.to_dict()
                for name, result in champion["validation_profiles"].items()
            },
            "feature_stats": {
                "mean": train_features.mean().to_dict(),
                "std": train_features.std(ddof=0).replace(0.0, 1.0).to_dict(),
            },
        },
    )
    test_prediction_frame["actual_class"] = y_test
    test_prediction_frame.to_csv(config.predictions_path, index=False)

    bundle = {
        "model_name": champion_name,
        "model": champion["estimator"],
        "feature_builder": feature_builder,
        "feature_columns": feature_names,
        "thresholds": {
            name: result.to_dict() for name, result in champion["validation_profiles"].items()
        },
        "feature_stats": {
            "mean": train_features.mean().to_dict(),
            "std": train_features.std(ddof=0).replace(0.0, 1.0).to_dict(),
        },
        "metadata": {
            "selection_metric": "validation balanced_f2",
            "raw_data_path": str(config.raw_data_path),
            "train_rows": len(splits.train),
            "validation_rows": len(splits.validation),
            "test_rows": len(splits.test),
            "raw_feature_columns": RAW_FEATURE_COLUMNS,
            "artifact_dir": str(config.artifact_dir),
        },
    }
    joblib.dump(bundle, config.model_bundle_path)

    plot_precision_recall_curve(
        y_true=pd.Series(y_test),
        scores=pd.Series(champion["test_scores"]),
        output_path=config.figures_dir / "precision_recall_curve.png",
    )
    plot_threshold_tradeoffs(
        champion["validation_threshold_curve"],
        config.figures_dir / "threshold_tradeoffs.png",
    )
    if not importance_frame.empty:
        plot_feature_importance(
            importance_frame,
            config.figures_dir / "feature_importance.png",
        )
    plot_drift_report(
        drift_frame,
        config.figures_dir / "drift_report.png",
    )

    metrics_payload = {
        "project_name": "Fraud Sentinel",
        "selection_metric": "Highest validation F2 at the balanced_f2 threshold profile",
        "dataset_summary": {
            "rows": len(raw_frame),
            "fraud_cases": int(raw_frame[TARGET_COLUMN].sum()),
            "fraud_rate": float(raw_frame[TARGET_COLUMN].mean()),
            "train_rows": len(splits.train),
            "validation_rows": len(splits.validation),
            "test_rows": len(splits.test),
        },
        "leaderboard": leaderboard.to_dict(orient="records"),
        "champion": {
            "model_name": champion_name,
            "description": champion["description"],
            "validation_profiles": {
                name: result.to_dict()
                for name, result in champion["validation_profiles"].items()
            },
            "test_profiles": {
                name: result.to_dict() for name, result in champion["test_profiles"].items()
            },
        },
        "models": {
            model_name: {
                "description": result["description"],
                "validation_profiles": {
                    name: threshold_result.to_dict()
                    for name, threshold_result in result["validation_profiles"].items()
                },
                "test_profiles": {
                    name: threshold_result.to_dict()
                    for name, threshold_result in result["test_profiles"].items()
                },
            }
            for model_name, result in candidate_results.items()
        },
        "top_feature_importance": importance_frame.head(12).to_dict(orient="records"),
        "top_drift_features": drift_frame.head(12).to_dict(orient="records"),
        "artifacts": {
            "model_bundle_path": str(config.model_bundle_path),
            "metrics_path": str(config.metrics_path),
            "leaderboard_path": str(config.leaderboard_path),
            "predictions_path": str(config.predictions_path),
            "drift_report_path": str(config.drift_report_path),
            "feature_importance_path": str(config.feature_importance_path),
        },
    }
    write_json(metrics_payload, config.metrics_path)
    return metrics_payload


def default_bundle_path(project_root: Path | None = None) -> Path:
    base = project_root or Path.cwd()
    return base / "artifacts" / "model_bundle.joblib"

