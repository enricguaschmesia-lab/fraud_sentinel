from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from fraud_sentinel.config import TrainingConfig
from fraud_sentinel.models import StrategySpec


@dataclass(frozen=True)
class TuningOutcome:
    strategy_name: str
    tuned: bool
    scoring: str
    best_score: float | None
    best_params: dict[str, Any]
    search_rows: int
    cv_folds: int


def tune_estimator(
    spec: StrategySpec,
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
) -> tuple[BaseEstimator, TuningOutcome, pd.DataFrame]:
    tuning_config = config.tuning
    if not tuning_config.enabled or not spec.enable_tuning or not spec.tuning_space:
        outcome = TuningOutcome(
            strategy_name=spec.name,
            tuned=False,
            scoring=tuning_config.scoring,
            best_score=None,
            best_params={},
            search_rows=len(X),
            cv_folds=tuning_config.cv_folds,
        )
        return estimator, outcome, pd.DataFrame()

    sample_X, sample_y = _sample_search_data(X, y, tuning_config.max_rows_for_search, config.random_state)
    cv = StratifiedKFold(
        n_splits=tuning_config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=spec.tuning_space,
        n_iter=tuning_config.search_iterations,
        scoring=tuning_config.scoring,
        cv=cv,
        random_state=config.random_state,
        n_jobs=1,
        refit=True,
    )
    search.fit(sample_X, sample_y)

    result_frame = pd.DataFrame(search.cv_results_)[
        ["rank_test_score", "mean_test_score", "std_test_score", "params"]
    ].sort_values("rank_test_score", kind="stable")
    result_frame.insert(0, "strategy_name", spec.name)
    outcome = TuningOutcome(
        strategy_name=spec.name,
        tuned=True,
        scoring=tuning_config.scoring,
        best_score=float(search.best_score_),
        best_params=dict(search.best_params_),
        search_rows=len(sample_X),
        cv_folds=tuning_config.cv_folds,
    )
    return search.best_estimator_, outcome, result_frame.reset_index(drop=True)


def _sample_search_data(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= max_rows:
        return X, y

    target_frame = X.assign(_target=y.to_numpy())
    sampled_parts: list[pd.DataFrame] = []
    for target_value, frame in target_frame.groupby("_target"):
        _ = target_value
        sampled_parts.append(
            frame.sample(
                n=max(1, int(round(max_rows * len(frame) / len(X)))),
                random_state=random_state,
            )
        )
    sampled = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=random_state)
    sampled_y = sampled.pop("_target").astype(int)
    return sampled, sampled_y
