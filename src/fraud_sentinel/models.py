from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_sentinel.config import TrainingConfig


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: BaseEstimator
    description: str
    fit_on_majority_only: bool = False


def build_candidate_models(config: TrainingConfig) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="logistic_regression",
            estimator=Pipeline(
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
            ),
            description="Linear benchmark with class weighting and calibrated-looking scores.",
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=config.random_forest_estimators,
                max_depth=14,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=config.random_state,
                n_jobs=1,
            ),
            description="Non-linear ensemble tuned for strong fraud-recall / false-positive balance.",
        ),
        ModelSpec(
            name="balanced_random_forest",
            estimator=BalancedRandomForestClassifier(
                n_estimators=config.balanced_random_forest_estimators,
                max_depth=12,
                random_state=config.random_state,
                n_jobs=1,
            ),
            description="Imbalance-aware challenger that resamples the minority class per tree.",
        ),
        ModelSpec(
            name="isolation_forest",
            estimator=IsolationForest(
                n_estimators=config.isolation_forest_estimators,
                contamination=0.002,
                random_state=config.random_state,
                n_jobs=1,
            ),
            description="Unsupervised anomaly-detection baseline trained only on legitimate traffic.",
            fit_on_majority_only=True,
        ),
    ]


def fit_model(spec: ModelSpec, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
    estimator = spec.estimator
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
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "named_steps") and hasattr(
        estimator.named_steps["classifier"], "coef_"
    ):
        importances = np.abs(estimator.named_steps["classifier"].coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance_frame = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False, kind="stable")
    return importance_frame.reset_index(drop=True)
