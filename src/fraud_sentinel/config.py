from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.64
    validation_fraction: float = 0.16
    test_fraction: float = 0.20

    def validate(self) -> None:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1.0, got {total:.6f}.")


@dataclass(frozen=True)
class ThresholdConfig:
    strict_alert_rate: float = 0.0005
    analyst_alert_rate: float = 0.0010
    aggressive_alert_rate: float = 0.0020
    threshold_grid_size: int = 300


@dataclass(frozen=True)
class BusinessConfig:
    review_cost: float = 2.5
    recovered_amount_factor: float = 0.85
    missed_fraud_penalty: float = 1.0


@dataclass(frozen=True)
class TuningConfig:
    enabled: bool = True
    cv_folds: int = 3
    search_iterations: int = 4
    scoring: str = "average_precision"
    max_rows_for_search: int = 60_000


@dataclass(frozen=True)
class DiagnosticsConfig:
    enabled: bool = True
    learning_curve_enabled: bool = True
    max_rows_for_learning_curve: int = 20_000
    max_rows_for_distribution_plots: int = 8_000
    top_feature_count: int = 6
    max_error_examples_per_profile: int = 10


@dataclass(frozen=True)
class TrainingConfig:
    raw_data_path: Path = Path("data/raw/creditcard.csv")
    processed_dir: Path = Path("data/processed")
    artifact_dir: Path = Path("artifacts")
    figures_dir: Path = Path("artifacts/figures")
    model_bundle_path: Path = Path("artifacts/model_bundle.joblib")
    metrics_path: Path = Path("artifacts/metrics.json")
    leaderboard_path: Path = Path("artifacts/leaderboard.csv")
    anomaly_leaderboard_path: Path = Path("artifacts/anomaly_leaderboard.csv")
    tuning_results_path: Path = Path("artifacts/tuning_results.csv")
    strategy_comparison_path: Path = Path("artifacts/strategy_comparison.csv")
    class_distribution_path: Path = Path("artifacts/class_distribution.csv")
    correlation_report_path: Path = Path("artifacts/correlation_summary.csv")
    predictions_path: Path = Path("artifacts/test_predictions.csv")
    drift_report_path: Path = Path("artifacts/drift_report.csv")
    feature_importance_path: Path = Path("artifacts/feature_importance.csv")
    error_analysis_path: Path = Path("artifacts/error_analysis.csv")
    random_state: int = 42
    random_forest_estimators: int = 200
    balanced_random_forest_estimators: int = 200
    isolation_forest_estimators: int = 200
    smote_random_forest_estimators: int = 140
    sample_rows: int | None = None
    candidate_strategies: tuple[str, ...] | None = None
    split: SplitConfig = field(default_factory=SplitConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    business: BusinessConfig = field(default_factory=BusinessConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    def ensure_directories(self) -> None:
        for directory in (self.processed_dir, self.artifact_dir, self.figures_dir):
            directory.mkdir(parents=True, exist_ok=True)
