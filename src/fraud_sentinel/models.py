from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import NearMiss
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_sentinel.config import TrainingConfig


StrategyBuilder = Callable[[TrainingConfig], BaseEstimator]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    family: str
    imbalance_strategy: str
    estimator_name: str
    description: str
    builder: StrategyBuilder
    is_supervised: bool = True
    fit_on_majority_only: bool = False
    tuning_space: dict[str, list[Any]] | None = None
    enable_tuning: bool = False


def build_candidate_models(config: TrainingConfig) -> list[StrategySpec]:
    strategies = [
        StrategySpec(
            name="weighted_logistic_regression",
            family="linear_model",
            imbalance_strategy="class_weight",
            estimator_name="logistic_regression",
            description="Weighted logistic regression baseline with standardized features.",
            builder=_build_weighted_logistic_regression,
            tuning_space={
                "classifier__penalty": ["l1", "l2"],
                "classifier__C": [0.1, 1.0, 5.0, 10.0],
            },
            enable_tuning=True,
        ),
        StrategySpec(
            name="random_forest",
            family="tree_ensemble",
            imbalance_strategy="class_weight",
            estimator_name="random_forest",
            description="Random forest with class-balanced bootstrapping.",
            builder=_build_random_forest,
            tuning_space={
                "n_estimators": [120, config.random_forest_estimators],
                "max_depth": [10, 14, None],
                "min_samples_leaf": [1, 2, 4],
            },
            enable_tuning=True,
        ),
        StrategySpec(
            name="balanced_random_forest",
            family="tree_ensemble",
            imbalance_strategy="balanced_bootstrap",
            estimator_name="balanced_random_forest",
            description="Balanced random forest that rebalances each tree during fitting.",
            builder=_build_balanced_random_forest,
            tuning_space={
                "n_estimators": [120, config.balanced_random_forest_estimators],
                "max_depth": [8, 12, None],
                "min_samples_leaf": [1, 2, 4],
            },
            enable_tuning=True,
        ),
        StrategySpec(
            name="smote_logistic_regression",
            family="linear_model",
            imbalance_strategy="smote",
            estimator_name="logistic_regression",
            description="SMOTE + logistic regression benchmark implemented as an imbalanced-learn pipeline.",
            builder=_build_smote_logistic_regression,
            tuning_space={
                "sampler__k_neighbors": [3, 5],
                "classifier__penalty": ["l1", "l2"],
                "classifier__C": [0.1, 1.0, 5.0, 10.0],
            },
            enable_tuning=True,
        ),
        StrategySpec(
            name="smote_random_forest",
            family="tree_ensemble",
            imbalance_strategy="smote",
            estimator_name="random_forest",
            description="SMOTE + random forest challenger to compare synthetic oversampling with weighted trees.",
            builder=_build_smote_random_forest,
        ),
        StrategySpec(
            name="nearmiss_logistic_regression",
            family="linear_model",
            imbalance_strategy="nearmiss",
            estimator_name="logistic_regression",
            description="NearMiss + logistic regression teaching baseline; operationally weaker but useful for comparison.",
            builder=_build_nearmiss_logistic_regression,
        ),
        StrategySpec(
            name="isolation_forest",
            family="anomaly_detection",
            imbalance_strategy="none",
            estimator_name="isolation_forest",
            description="Unsupervised anomaly-detection baseline trained only on legitimate traffic.",
            builder=_build_isolation_forest,
            is_supervised=False,
            fit_on_majority_only=True,
        ),
    ]
    if config.candidate_strategies is None:
        return strategies
    selected = set(config.candidate_strategies)
    return [strategy for strategy in strategies if strategy.name in selected]


def fit_model(spec: StrategySpec, estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
    if spec.fit_on_majority_only:
        estimator.fit(X.loc[y == 0])
    else:
        estimator.fit(X, y)
    return estimator


def score_model(estimator: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    else:
        raw_predictions = estimator.predict(X)
        scores = np.asarray(raw_predictions, dtype=float)
    return np.asarray(scores, dtype=float)


def extract_feature_importance(
    estimator: BaseEstimator,
    feature_names: list[str],
) -> pd.DataFrame:
    resolved_estimator = _final_estimator(estimator)
    if hasattr(resolved_estimator, "feature_importances_"):
        importances = np.asarray(resolved_estimator.feature_importances_, dtype=float)
    elif hasattr(resolved_estimator, "coef_"):
        importances = np.abs(resolved_estimator.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance_frame = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False, kind="stable")
    return importance_frame.reset_index(drop=True)


def resolved_estimator_name(estimator: BaseEstimator) -> str:
    return _final_estimator(estimator).__class__.__name__


def _final_estimator(estimator: BaseEstimator) -> BaseEstimator:
    if hasattr(estimator, "named_steps"):
        return next(reversed(estimator.named_steps.values()))
    if hasattr(estimator, "steps"):
        return estimator.steps[-1][1]
    return estimator


def _build_weighted_logistic_regression(config: TrainingConfig) -> BaseEstimator:
    _ = config
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _build_random_forest(config: TrainingConfig) -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=config.random_forest_estimators,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=config.random_state,
        n_jobs=1,
    )


def _build_balanced_random_forest(config: TrainingConfig) -> BaseEstimator:
    return BalancedRandomForestClassifier(
        n_estimators=config.balanced_random_forest_estimators,
        max_depth=12,
        min_samples_leaf=2,
        random_state=config.random_state,
        n_jobs=1,
    )


def _build_smote_logistic_regression(config: TrainingConfig) -> BaseEstimator:
    return ImbPipeline(
        steps=[
            ("sampler", SMOTE(random_state=config.random_state, k_neighbors=5)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _build_smote_random_forest(config: TrainingConfig) -> BaseEstimator:
    return ImbPipeline(
        steps=[
            ("sampler", SMOTE(random_state=config.random_state, k_neighbors=5)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=config.smote_random_forest_estimators,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=config.random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _build_nearmiss_logistic_regression(config: TrainingConfig) -> BaseEstimator:
    return ImbPipeline(
        steps=[
            ("sampler", NearMiss(version=1)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    solver="liblinear",
                ),
            ),
        ]
    )


def _build_isolation_forest(config: TrainingConfig) -> BaseEstimator:
    return IsolationForest(
        n_estimators=config.isolation_forest_estimators,
        contamination=0.002,
        random_state=config.random_state,
        n_jobs=1,
    )
