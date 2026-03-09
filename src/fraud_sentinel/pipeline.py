from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from fraud_sentinel.config import TrainingConfig
from fraud_sentinel.data import RAW_FEATURE_COLUMNS, TARGET_COLUMN, load_dataset, split_dataset
from fraud_sentinel.evaluation import (
    build_error_analysis,
    class_distribution_summary,
    compute_threshold_profiles,
    correlation_summary,
    drift_report,
    evaluate_fixed_thresholds,
    leaderboard_row,
    ranking_metrics,
)
from fraud_sentinel.features import FraudFeatureBuilder
from fraud_sentinel.inference import score_transactions
from fraud_sentinel.models import (
    build_candidate_models,
    extract_feature_importance,
    fit_model,
    resolved_estimator_name,
    score_model,
)
from fraud_sentinel.reporting import (
    plot_calibration_curve,
    plot_class_score_distribution,
    plot_confusion_matrix_profiles,
    plot_correlation_summary,
    plot_drift_report,
    plot_feature_distribution_comparison,
    plot_feature_importance,
    plot_learning_curve_summary,
    plot_precision_recall_curve,
    plot_threshold_tradeoffs,
    write_json,
    write_table,
)
from fraud_sentinel.tuning import tune_estimator


def run_training(config: TrainingConfig | None = None) -> dict[str, Any]:
    config = config or TrainingConfig()
    config.ensure_directories()

    raw_frame = load_dataset(config.raw_data_path, sample_rows=config.sample_rows)
    splits = split_dataset(raw_frame, config.split)
    split_map = {
        "full_dataset": raw_frame,
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }

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

    class_distribution = class_distribution_summary(split_map, TARGET_COLUMN)
    correlation_report = correlation_summary(
        train_features.assign(Class=y_train.to_numpy()),
        target_column="Class",
    )
    write_table(class_distribution, config.class_distribution_path)
    write_table(correlation_report, config.correlation_report_path)

    candidate_results: dict[str, dict[str, Any]] = {}
    supervised_rows: list[dict[str, Any]] = []
    anomaly_rows: list[dict[str, Any]] = []
    tuning_frames: list[pd.DataFrame] = []

    for spec in build_candidate_models(config):
        estimator = spec.builder(config)
        tuned_estimator, tuning_outcome, tuning_frame = tune_estimator(
            spec,
            estimator,
            train_features,
            y_train,
            config,
        )
        if not tuning_frame.empty:
            tuning_frames.append(tuning_frame)

        estimator = fit_model(spec, tuned_estimator, train_features, y_train)

        train_scores = score_model(estimator, train_features)
        validation_scores = score_model(estimator, validation_features)
        test_scores = score_model(estimator, test_features)

        train_metrics = ranking_metrics(y_train.to_numpy(), train_scores)
        validation_metrics = ranking_metrics(y_validation, validation_scores)
        test_metrics = ranking_metrics(y_test, test_scores)

        validation_profiles, threshold_frame = compute_threshold_profiles(
            y_true=y_validation,
            scores=validation_scores,
            amounts=validation_amounts,
            threshold_config=config.thresholds,
            business_config=config.business,
        )
        test_profiles = evaluate_fixed_thresholds(
            y_true=y_test,
            scores=test_scores,
            amounts=test_amounts,
            thresholds=validation_profiles,
            business_config=config.business,
        )

        row = leaderboard_row(
            model_name=spec.name,
            description=spec.description,
            family=spec.family,
            imbalance_strategy=spec.imbalance_strategy,
            estimator_name=resolved_estimator_name(estimator),
            is_supervised=spec.is_supervised,
            train_metrics=train_metrics,
            validation_profiles=validation_profiles,
            validation_metrics=validation_metrics,
            test_profiles=test_profiles,
            tuning_outcome={
                "tuned": tuning_outcome.tuned,
                "best_score": tuning_outcome.best_score,
            },
        )
        if spec.is_supervised:
            supervised_rows.append(row)
        else:
            anomaly_rows.append(row)

        candidate_results[spec.name] = {
            "spec": spec,
            "estimator": estimator,
            "train_scores": train_scores,
            "validation_scores": validation_scores,
            "test_scores": test_scores,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "validation_profiles": validation_profiles,
            "test_profiles": test_profiles,
            "validation_threshold_curve": threshold_frame,
            "tuning_outcome": tuning_outcome,
        }

    leaderboard = pd.DataFrame(supervised_rows).sort_values(
        ["selection_score", "validation_average_precision"],
        ascending=False,
        kind="stable",
    )
    anomaly_leaderboard = pd.DataFrame(anomaly_rows).sort_values(
        ["validation_average_precision", "validation_roc_auc"],
        ascending=False,
        kind="stable",
    )
    tuning_results = (
        pd.concat(tuning_frames, ignore_index=True)
        if tuning_frames
        else pd.DataFrame(columns=["strategy_name", "rank_test_score", "mean_test_score", "std_test_score", "params"])
    )
    strategy_comparison = leaderboard[
        [
            "model_name",
            "family",
            "imbalance_strategy",
            "estimator_name",
            "validation_average_precision",
            "validation_f2",
            "test_average_precision",
            "test_f2",
            "strict_queue_precision",
            "strict_queue_recall",
            "aggressive_queue_precision",
            "aggressive_queue_recall",
            "average_precision_gap",
            "tuned",
        ]
    ].copy()
    write_table(leaderboard, config.leaderboard_path)
    write_table(anomaly_leaderboard, config.anomaly_leaderboard_path)
    write_table(tuning_results, config.tuning_results_path)
    write_table(strategy_comparison, config.strategy_comparison_path)

    champion_name = str(leaderboard.iloc[0]["model_name"])
    champion = candidate_results[champion_name]
    best_ranking_name = str(
        leaderboard.sort_values("validation_average_precision", ascending=False, kind="stable").iloc[0]["model_name"]
    )
    best_ranking_model = candidate_results[best_ranking_name]

    importance_frame = extract_feature_importance(champion["estimator"], feature_names)
    drift_frame = drift_report(train_features, test_features)
    write_table(importance_frame, config.feature_importance_path)
    write_table(drift_frame, config.drift_report_path)

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
            "family": champion["spec"].family,
            "imbalance_strategy": champion["spec"].imbalance_strategy,
            "estimator_name": resolved_estimator_name(champion["estimator"]),
        },
    }
    joblib.dump(bundle, config.model_bundle_path)

    test_prediction_frame = score_transactions(
        splits.test[RAW_FEATURE_COLUMNS],
        bundle=bundle,
    )
    test_prediction_frame["actual_class"] = y_test
    test_prediction_frame.to_csv(config.predictions_path, index=False)

    error_analysis = build_error_analysis(
        test_prediction_frame,
        champion["test_profiles"],
        config.diagnostics.max_error_examples_per_profile,
    )
    write_table(error_analysis, config.error_analysis_path)

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
    plot_drift_report(drift_frame, config.figures_dir / "drift_report.png")

    diagnostics_summary: dict[str, Any] = {
        "class_distribution_path": str(config.class_distribution_path),
        "correlation_report_path": str(config.correlation_report_path),
        "strategy_comparison_path": str(config.strategy_comparison_path),
        "error_analysis_path": str(config.error_analysis_path),
    }
    if config.diagnostics.enabled:
        calibration_summary = plot_calibration_curve(
            y_true=y_test,
            scores=champion["test_scores"],
            output_path=config.figures_dir / "calibration_curve.png",
        )
        plot_class_score_distribution(
            y_true=y_test,
            scores=champion["test_scores"],
            output_path=config.figures_dir / "score_distribution.png",
        )
        plot_confusion_matrix_profiles(
            y_true=y_test,
            scores=champion["test_scores"],
            thresholds={name: result.to_dict() for name, result in champion["test_profiles"].items()},
            output_path=config.figures_dir / "confusion_matrix_profiles.png",
        )
        top_feature_names = correlation_report["feature"].head(config.diagnostics.top_feature_count).tolist()
        distribution_sample = train_features.assign(Class=y_train.to_numpy())
        if len(distribution_sample) > config.diagnostics.max_rows_for_distribution_plots:
            sampled_groups: list[pd.DataFrame] = []
            for _, frame in distribution_sample.groupby("Class"):
                sampled_groups.append(
                    frame.sample(
                        n=min(
                            len(frame),
                            max(1, config.diagnostics.max_rows_for_distribution_plots // 2),
                        ),
                        random_state=config.random_state,
                    )
                )
            distribution_sample = pd.concat(sampled_groups, axis=0).sample(
                frac=1.0,
                random_state=config.random_state,
            )
        plot_feature_distribution_comparison(
            feature_frame=distribution_sample.drop(columns="Class"),
            y_true=distribution_sample["Class"],
            feature_names=top_feature_names,
            output_path=config.figures_dir / "feature_distribution_comparison.png",
        )
        plot_correlation_summary(
            correlation_report,
            config.figures_dir / "correlation_summary.png",
        )
        diagnostics_summary["calibration"] = calibration_summary
        diagnostics_summary["generated_figures"] = [
            "precision_recall_curve.png",
            "threshold_tradeoffs.png",
            "feature_importance.png",
            "drift_report.png",
            "calibration_curve.png",
            "score_distribution.png",
            "confusion_matrix_profiles.png",
            "feature_distribution_comparison.png",
            "correlation_summary.png",
        ]
        if config.diagnostics.learning_curve_enabled:
            try:
                plot_learning_curve_summary(
                    champion["estimator"],
                    train_features,
                    y_train,
                    config.figures_dir / "learning_curve.png",
                    random_state=config.random_state,
                )
                diagnostics_summary["generated_figures"].append("learning_curve.png")
            except ValueError:
                diagnostics_summary["learning_curve_warning"] = "Learning curve skipped because the sampled folds were not feasible."

    model_card = {
        "dataset": "ULB credit-card fraud dataset via TensorFlow-hosted CSV mirror",
        "split_method": "chronological 64/16/20 split by Time",
        "imbalance_strategy": champion["spec"].imbalance_strategy,
        "champion_rationale": (
            "Selected as the highest validation balanced_f2 model while preserving strong precision at low alert rates."
        ),
        "limitations": [
            "Public benchmark features V1-V28 are anonymized PCA-like components.",
            "The deployed reason codes are heuristic z-score explanations, not causal explanations.",
            "Resampling challengers are included for comparison, but production decisions still rely on chronological holdout behavior.",
        ],
    }
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
        "anomaly_leaderboard": anomaly_leaderboard.to_dict(orient="records"),
        "best_ranking_model": {
            "model_name": best_ranking_name,
            "validation_average_precision": best_ranking_model["validation_metrics"]["average_precision"],
            "test_average_precision": best_ranking_model["test_metrics"]["average_precision"],
        },
        "champion": {
            "model_name": champion_name,
            "description": champion["spec"].description,
            "family": champion["spec"].family,
            "imbalance_strategy": champion["spec"].imbalance_strategy,
            "estimator_name": resolved_estimator_name(champion["estimator"]),
            "validation_profiles": {
                name: result.to_dict()
                for name, result in champion["validation_profiles"].items()
            },
            "test_profiles": {
                name: result.to_dict() for name, result in champion["test_profiles"].items()
            },
            "train_metrics": champion["train_metrics"],
            "validation_metrics": champion["validation_metrics"],
            "test_metrics": champion["test_metrics"],
            "tuning": champion["tuning_outcome"].__dict__,
        },
        "models": {
            model_name: {
                "description": result["spec"].description,
                "family": result["spec"].family,
                "imbalance_strategy": result["spec"].imbalance_strategy,
                "estimator_name": resolved_estimator_name(result["estimator"]),
                "is_supervised": result["spec"].is_supervised,
                "train_metrics": result["train_metrics"],
                "validation_metrics": result["validation_metrics"],
                "test_metrics": result["test_metrics"],
                "validation_profiles": {
                    name: threshold_result.to_dict()
                    for name, threshold_result in result["validation_profiles"].items()
                },
                "test_profiles": {
                    name: threshold_result.to_dict()
                    for name, threshold_result in result["test_profiles"].items()
                },
                "tuning": result["tuning_outcome"].__dict__,
            }
            for model_name, result in candidate_results.items()
        },
        "resampling_strategy_comparison": strategy_comparison.to_dict(orient="records"),
        "tuning_results_preview": tuning_results.head(20).to_dict(orient="records"),
        "class_distribution": class_distribution.to_dict(orient="records"),
        "correlation_summary": correlation_report.head(20).to_dict(orient="records"),
        "top_feature_importance": importance_frame.head(12).to_dict(orient="records"),
        "top_drift_features": drift_frame.head(12).to_dict(orient="records"),
        "error_analysis_preview": error_analysis.head(20).to_dict(orient="records"),
        "diagnostics": diagnostics_summary,
        "model_card": model_card,
        "artifacts": {
            "model_bundle_path": str(config.model_bundle_path),
            "metrics_path": str(config.metrics_path),
            "leaderboard_path": str(config.leaderboard_path),
            "anomaly_leaderboard_path": str(config.anomaly_leaderboard_path),
            "tuning_results_path": str(config.tuning_results_path),
            "strategy_comparison_path": str(config.strategy_comparison_path),
            "class_distribution_path": str(config.class_distribution_path),
            "correlation_report_path": str(config.correlation_report_path),
            "predictions_path": str(config.predictions_path),
            "drift_report_path": str(config.drift_report_path),
            "feature_importance_path": str(config.feature_importance_path),
            "error_analysis_path": str(config.error_analysis_path),
        },
    }
    write_json(metrics_payload, config.metrics_path)
    return metrics_payload


def default_bundle_path(project_root: Path | None = None) -> Path:
    base = project_root or Path.cwd()
    return base / "artifacts" / "model_bundle.joblib"
