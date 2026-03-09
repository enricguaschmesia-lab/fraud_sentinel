from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, learning_curve


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_table(frame: pd.DataFrame, output_path: Path) -> None:
    frame.to_csv(output_path, index=False)


def plot_precision_recall_curve(
    y_true: pd.Series,
    scores: pd.Series,
    output_path: Path,
) -> None:
    display = PrecisionRecallDisplay.from_predictions(y_true, scores)
    display.ax_.set_title("Champion Precision-Recall Curve")
    display.ax_.grid(alpha=0.2)
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=160)
    plt.close(display.figure_)


def plot_threshold_tradeoffs(threshold_frame: pd.DataFrame, output_path: Path) -> None:
    plot_frame = threshold_frame.sort_values("threshold", ascending=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(plot_frame["threshold"], plot_frame["precision"], label="Precision", color="#CC5500")
    axis.plot(plot_frame["threshold"], plot_frame["recall"], label="Recall", color="#005A9C")
    axis.plot(plot_frame["threshold"], plot_frame["f2"], label="F2", color="#0B6E4F")
    axis.set_title("Validation Threshold Trade-offs")
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Metric value")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_feature_importance(importance_frame: pd.DataFrame, output_path: Path) -> None:
    top_features = importance_frame.head(12).iloc[::-1]
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(top_features["feature"], top_features["importance"], color="#0B6E4F")
    axis.set_title("Champion Feature Importance")
    axis.set_xlabel("Importance")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_drift_report(drift_frame: pd.DataFrame, output_path: Path) -> None:
    top_drift = drift_frame.head(12).iloc[::-1]
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(top_drift["feature"], top_drift["psi"], color="#005A9C")
    axis.set_title("Top Train/Test Drift Features")
    axis.set_xlabel("Population Stability Index")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_class_score_distribution(
    y_true: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(scores[y_true == 0], bins=50, alpha=0.65, label="Legitimate", color="#1e4e79", density=True)
    axis.hist(scores[y_true == 1], bins=50, alpha=0.65, label="Fraud", color="#b33a3a", density=True)
    axis.set_title("Champion Score Distribution by Class")
    axis.set_xlabel("Risk score")
    axis.set_ylabel("Density")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_calibration_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
) -> dict[str, float]:
    clipped_scores = np.clip(scores, 0.0, 1.0)
    prob_true, prob_pred = calibration_curve(y_true, clipped_scores, n_bins=10, strategy="quantile")

    figure, axis = plt.subplots(figsize=(6.5, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#666666", label="Perfect calibration")
    axis.plot(prob_pred, prob_true, marker="o", color="#0B6E4F", label="Champion")
    axis.set_title("Calibration Curve")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed fraud rate")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    calibration_error = float(np.mean(np.abs(prob_true - prob_pred))) if len(prob_true) else 0.0
    return {
        "bin_count": int(len(prob_true)),
        "mean_absolute_calibration_error": calibration_error,
    }


def plot_confusion_matrix_profiles(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    profile_names = list(thresholds.keys())
    figure, axes = plt.subplots(1, len(profile_names), figsize=(5 * len(profile_names), 4.5))
    if len(profile_names) == 1:
        axes = [axes]

    for axis, profile_name in zip(axes, profile_names):
        threshold = thresholds[profile_name]["threshold"]
        predictions = (scores >= threshold).astype(int)
        matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
        display = ConfusionMatrixDisplay(matrix, display_labels=["Legitimate", "Fraud"])
        display.plot(ax=axis, colorbar=False, cmap="Blues")
        axis.set_title(profile_name.replace("_", " ").title())

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_feature_distribution_comparison(
    feature_frame: pd.DataFrame,
    y_true: pd.Series,
    feature_names: list[str],
    output_path: Path,
) -> None:
    fraud_mask = y_true.to_numpy() == 1
    figure, axes = plt.subplots(len(feature_names), 1, figsize=(8, 3.5 * len(feature_names)))
    if len(feature_names) == 1:
        axes = [axes]

    for axis, feature_name in zip(axes, feature_names):
        axis.hist(
            feature_frame.loc[~fraud_mask, feature_name],
            bins=40,
            density=True,
            alpha=0.55,
            color="#1e4e79",
            label="Legitimate",
        )
        axis.hist(
            feature_frame.loc[fraud_mask, feature_name],
            bins=40,
            density=True,
            alpha=0.55,
            color="#b33a3a",
            label="Fraud",
        )
        axis.set_title(f"Feature Distribution: {feature_name}")
        axis.grid(alpha=0.2)
        axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_correlation_summary(correlation_frame: pd.DataFrame, output_path: Path) -> None:
    top_features = correlation_frame.head(12).iloc[::-1]
    figure, axis = plt.subplots(figsize=(8, 6))
    colors = np.where(top_features["correlation"] >= 0, "#CC5500", "#1e4e79")
    axis.barh(top_features["feature"], top_features["correlation"], color=colors)
    axis.set_title("Top Feature Correlations with Fraud Label")
    axis.set_xlabel("Correlation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_learning_curve_summary(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
    random_state: int,
) -> None:
    sample_frame = X.assign(_target=y.to_numpy())
    if len(sample_frame) > 20_000:
        sampled_parts: list[pd.DataFrame] = []
        for _, frame in sample_frame.groupby("_target"):
            sampled_parts.append(
                frame.sample(n=min(len(frame), 10_000), random_state=random_state)
            )
        sample_frame = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=random_state)
    sample_y = sample_frame.pop("_target")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator,
        sample_frame,
        sample_y,
        cv=cv,
        scoring="average_precision",
        n_jobs=1,
        train_sizes=np.linspace(0.2, 1.0, 5),
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Train AP", color="#CC5500")
    axis.plot(
        train_sizes,
        validation_scores.mean(axis=1),
        marker="o",
        label="Cross-validation AP",
        color="#0B6E4F",
    )
    axis.fill_between(
        train_sizes,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.12,
        color="#CC5500",
    )
    axis.fill_between(
        train_sizes,
        validation_scores.mean(axis=1) - validation_scores.std(axis=1),
        validation_scores.mean(axis=1) + validation_scores.std(axis=1),
        alpha=0.12,
        color="#0B6E4F",
    )
    axis.set_title("Champion Learning Curve")
    axis.set_xlabel("Training rows")
    axis.set_ylabel("Average precision")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
