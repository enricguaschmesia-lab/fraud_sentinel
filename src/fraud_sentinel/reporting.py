from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_table(frame: pd.DataFrame, output_path: Path) -> None:
    frame.to_csv(output_path, index=False)


def plot_precision_recall_curve(
    y_true: pd.Series,
    scores: pd.Series,
    output_path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall, precision, linewidth=2.0, color="#0B6E4F")
    axis.set_title("Champion Precision-Recall Curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


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
