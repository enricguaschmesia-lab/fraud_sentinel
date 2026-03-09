from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from fraud_sentinel.inference import load_bundle, score_transactions


PROJECT_ROOT = Path.cwd()
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"
BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "model_bundle.joblib"
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "sample_transactions.csv"


def load_metrics() -> dict[str, object]:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "Run `python -m fraud_sentinel.cli train` before opening the dashboard."
        )
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="Fraud Sentinel", page_icon="🛡️", layout="wide")
    _inject_styles()

    metrics = load_metrics()
    bundle = load_bundle(BUNDLE_PATH)
    threshold_names = list(bundle["thresholds"].keys())
    selected_threshold = st.sidebar.selectbox(
        "Threshold profile",
        threshold_names,
        index=threshold_names.index("balanced_f2") if "balanced_f2" in threshold_names else 0,
    )

    champion_name = bundle["model_name"]
    champion_metrics = metrics["champion"]["test_profiles"][selected_threshold]

    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Imbalanced ML Benchmark</div>
          <h1>Fraud Sentinel</h1>
          <p>Operational fraud detection with imbalance-strategy benchmarking, threshold-aware triage, and generated diagnostics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Champion", champion_name)
    metric_columns[1].metric("Average Precision", f"{champion_metrics['average_precision']:.3f}")
    metric_columns[2].metric("Precision", f"{champion_metrics['precision']:.3f}")
    metric_columns[3].metric("Recall", f"{champion_metrics['recall']:.3f}")

    overview_left, overview_right = st.columns([1.3, 1.0])
    with overview_left:
        st.subheader("Supervised Leaderboard")
        leaderboard = pd.DataFrame(metrics["leaderboard"])
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)

    with overview_right:
        st.subheader("Threshold profile")
        threshold_frame = pd.DataFrame(bundle["thresholds"]).T.reset_index(names="profile")
        st.dataframe(threshold_frame, use_container_width=True, hide_index=True)

    resampling_left, resampling_right = st.columns([1.0, 1.0])
    with resampling_left:
        st.subheader("Resampling Strategy Comparison")
        comparison = pd.DataFrame(metrics.get("resampling_strategy_comparison", []))
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    with resampling_right:
        st.subheader("Model Card")
        model_card = metrics.get("model_card", {})
        st.json(model_card)

    anomaly_rows = metrics.get("anomaly_leaderboard", [])
    if anomaly_rows:
        st.subheader("Anomaly Baselines")
        st.dataframe(pd.DataFrame(anomaly_rows), use_container_width=True, hide_index=True)

    plots_left, plots_right = st.columns(2)
    with plots_left:
        _show_plot("artifacts/figures/precision_recall_curve.png", "Precision-Recall")
        _show_plot("artifacts/figures/feature_importance.png", "Feature Importance")
        _show_plot("artifacts/figures/calibration_curve.png", "Calibration")
        _show_plot("artifacts/figures/correlation_summary.png", "Correlation Summary")
    with plots_right:
        _show_plot("artifacts/figures/threshold_tradeoffs.png", "Threshold Trade-offs")
        _show_plot("artifacts/figures/drift_report.png", "Train/Test Drift")
        _show_plot("artifacts/figures/score_distribution.png", "Score Distribution")
        _show_plot("artifacts/figures/confusion_matrix_profiles.png", "Threshold Confusion Matrices")

    extra_left, extra_right = st.columns(2)
    with extra_left:
        _show_plot("artifacts/figures/feature_distribution_comparison.png", "Feature Distributions")
    with extra_right:
        _show_plot("artifacts/figures/learning_curve.png", "Learning Curve")

    diagnostics = metrics.get("diagnostics", {})
    if diagnostics:
        st.subheader("Diagnostics Summary")
        st.json(diagnostics)

    error_preview = pd.DataFrame(metrics.get("error_analysis_preview", []))
    if not error_preview.empty:
        st.subheader("Error Analysis Preview")
        st.dataframe(error_preview, use_container_width=True, hide_index=True)

    st.subheader("Batch scoring")
    uploaded_file = st.file_uploader("Upload CSV with raw transaction columns", type=["csv"])
    source_path = None
    if uploaded_file is not None:
        input_frame = pd.read_csv(uploaded_file)
    elif EXAMPLE_PATH.exists():
        source_path = EXAMPLE_PATH
        input_frame = pd.read_csv(EXAMPLE_PATH)
        st.caption(f"Using example data from `{EXAMPLE_PATH}`.")
    else:
        input_frame = None

    if input_frame is not None:
        scored = score_transactions(input_frame, bundle, threshold_profile=selected_threshold)
        st.dataframe(scored.head(50), use_container_width=True)
        st.download_button(
            label="Download scored CSV",
            data=scored.to_csv(index=False),
            file_name="fraud_sentinel_predictions.csv",
            mime="text/csv",
        )
    elif source_path is None:
        st.info("Upload a CSV or generate the example files by running training once.")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background:
              radial-gradient(circle at top left, rgba(11, 110, 79, 0.15), transparent 30%),
              linear-gradient(180deg, #f7f5ef 0%, #eef3f2 100%);
          }
          .hero {
            background: linear-gradient(135deg, #0b6e4f 0%, #1e4e79 100%);
            color: white;
            padding: 1.75rem 2rem;
            border-radius: 1.2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 18px 40px rgba(16, 41, 43, 0.12);
          }
          .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.75rem;
            opacity: 0.8;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _show_plot(path_text: str, title: str) -> None:
    path = PROJECT_ROOT / path_text
    st.subheader(title)
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.info(f"{path_text} will appear after training.")


if __name__ == "__main__":
    main()
