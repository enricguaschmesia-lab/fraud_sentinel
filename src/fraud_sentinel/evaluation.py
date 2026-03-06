from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from fraud_sentinel.config import BusinessConfig, ThresholdConfig


@dataclass(frozen=True)
class ThresholdResult:
    name: str
    threshold: float
    alert_count: int
    alert_rate: float
    precision: float
    recall: float
    f1: float
    f2: float
    average_precision: float
    roc_auc: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    captured_fraud_amount: float
    missed_fraud_amount: float
    review_cost_total: float
    business_score: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def compute_threshold_profiles(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    threshold_config: ThresholdConfig,
    business_config: BusinessConfig,
) -> tuple[dict[str, ThresholdResult], pd.DataFrame]:
    threshold_grid = _build_threshold_grid(scores, threshold_config.threshold_grid_size)
    evaluations = [
        _evaluate_threshold(
            threshold=float(threshold),
            y_true=y_true,
            scores=scores,
            amounts=amounts,
            business_config=business_config,
        )
        for threshold in threshold_grid
    ]
    threshold_frame = pd.DataFrame([evaluation.to_dict() for evaluation in evaluations])

    profiles: dict[str, ThresholdResult] = {}
    profiles["balanced_f2"] = _best_row(threshold_frame, "f2")
    profiles["business"] = _best_row(threshold_frame, "business_score")
    profiles["strict_queue"] = _queue_row(threshold_frame, threshold_config.strict_alert_rate)
    profiles["analyst_queue"] = _queue_row(threshold_frame, threshold_config.analyst_alert_rate)
    profiles["aggressive_queue"] = _queue_row(
        threshold_frame, threshold_config.aggressive_alert_rate
    )
    return profiles, threshold_frame


def evaluate_fixed_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    thresholds: dict[str, ThresholdResult],
    business_config: BusinessConfig,
) -> dict[str, ThresholdResult]:
    return {
        name: _evaluate_threshold(
            threshold=result.threshold,
            y_true=y_true,
            scores=scores,
            amounts=amounts,
            business_config=business_config,
            name=name,
        )
        for name, result in thresholds.items()
    }


def drift_report(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for column in train_frame.columns:
        psi = _population_stability_index(train_frame[column].to_numpy(), test_frame[column].to_numpy())
        rows.append({"feature": column, "psi": psi})

    report = pd.DataFrame(rows).sort_values("psi", ascending=False, kind="stable")
    return report.reset_index(drop=True)


def leaderboard_row(
    model_name: str,
    description: str,
    validation_profiles: dict[str, ThresholdResult],
    test_profiles: dict[str, ThresholdResult],
) -> dict[str, float | int | str]:
    validation_balanced = validation_profiles["balanced_f2"]
    test_balanced = test_profiles["balanced_f2"]
    return {
        "model_name": model_name,
        "description": description,
        "validation_average_precision": validation_balanced.average_precision,
        "validation_roc_auc": validation_balanced.roc_auc,
        "validation_f2": validation_balanced.f2,
        "validation_recall": validation_balanced.recall,
        "validation_precision": validation_balanced.precision,
        "test_average_precision": test_balanced.average_precision,
        "test_roc_auc": test_balanced.roc_auc,
        "test_f2": test_balanced.f2,
        "test_recall": test_balanced.recall,
        "test_precision": test_balanced.precision,
        "selection_score": validation_balanced.f2,
    }


def _build_threshold_grid(scores: np.ndarray, grid_size: int) -> np.ndarray:
    coarse_points = max(grid_size // 2, 50)
    fine_points = max(grid_size // 4, 25)
    ultra_fine_points = max(grid_size - coarse_points - fine_points, 25)
    quantiles = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 0.99, coarse_points, endpoint=False),
                np.linspace(0.99, 0.999, fine_points, endpoint=False),
                np.linspace(0.999, 0.999995, ultra_fine_points),
                np.array([0.999999, 1.0]),
            ]
        )
    )
    quantile_thresholds = np.quantile(scores, quantiles)
    unique_scores = np.unique(scores)
    tail_count = min(max(grid_size, 200), len(unique_scores))
    tail_thresholds = unique_scores[-tail_count:]
    thresholds = np.concatenate([quantile_thresholds, tail_thresholds])
    thresholds = np.unique(np.clip(thresholds, float(scores.min()), float(scores.max())))
    return thresholds[::-1]


def _evaluate_threshold(
    threshold: float,
    y_true: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    business_config: BusinessConfig,
    name: str = "candidate",
) -> ThresholdResult:
    predictions = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )
    f2 = _f_beta_score(precision, recall, beta=2.0)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    captured_amount = float(amounts[(y_true == 1) & predictions].sum())
    missed_amount = float(amounts[(y_true == 1) & (~predictions)].sum())
    alert_count = int(predictions.sum())
    review_cost_total = business_config.review_cost * alert_count
    business_score = (
        business_config.recovered_amount_factor * captured_amount
        - business_config.missed_fraud_penalty * missed_amount
        - review_cost_total
    )

    return ThresholdResult(
        name=name,
        threshold=float(threshold),
        alert_count=alert_count,
        alert_rate=alert_count / len(y_true),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        f2=float(f2),
        average_precision=float(average_precision_score(y_true, scores)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        true_positives=int(tp),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_negatives=int(tn),
        captured_fraud_amount=captured_amount,
        missed_fraud_amount=missed_amount,
        review_cost_total=float(review_cost_total),
        business_score=float(business_score),
    )


def _best_row(threshold_frame: pd.DataFrame, metric_name: str) -> ThresholdResult:
    row = threshold_frame.sort_values(metric_name, ascending=False, kind="stable").iloc[0]
    return ThresholdResult(**row.to_dict())


def _queue_row(threshold_frame: pd.DataFrame, alert_rate_limit: float) -> ThresholdResult:
    queue_rows = threshold_frame.loc[threshold_frame["alert_rate"] <= alert_rate_limit]
    if queue_rows.empty:
        queue_rows = threshold_frame.nsmallest(1, "alert_rate")
    row = queue_rows.sort_values(["recall", "precision"], ascending=False, kind="stable").iloc[0]
    return ThresholdResult(**row.to_dict())


def _f_beta_score(precision: float, recall: float, beta: float) -> float:
    beta_sq = beta**2
    denominator = beta_sq * precision + recall
    if denominator == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denominator


def _population_stability_index(train_values: np.ndarray, test_values: np.ndarray) -> float:
    quantiles = np.unique(np.quantile(train_values, np.linspace(0.0, 1.0, 11)))
    if len(quantiles) < 3:
        return 0.0

    bins = np.array(quantiles, dtype=float)
    bins[0] = -np.inf
    bins[-1] = np.inf

    train_hist, _ = np.histogram(train_values, bins=bins)
    test_hist, _ = np.histogram(test_values, bins=bins)

    train_pct = np.clip(train_hist / max(train_hist.sum(), 1), 1e-6, None)
    test_pct = np.clip(test_hist / max(test_hist.sum(), 1), 1e-6, None)
    return float(np.sum((test_pct - train_pct) * np.log(test_pct / train_pct)))
