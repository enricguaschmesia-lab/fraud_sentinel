from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from fraud_sentinel.config import TrainingConfig
from fraud_sentinel.inference import load_bundle, score_transactions
from fraud_sentinel.pipeline import run_training


app = typer.Typer(help="Fraud Sentinel training and inference CLI.")


@app.command()
def train(
    raw_data_path: Path = typer.Option(
        Path("data/raw/creditcard.csv"),
        help="CSV path for the source fraud dataset.",
    ),
    sample_rows: int | None = typer.Option(
        None,
        help="Optional row cap for quick experiments.",
    ),
    random_state: int = typer.Option(42, help="Random seed for the model pipelines."),
) -> None:
    config = TrainingConfig(
        raw_data_path=raw_data_path,
        sample_rows=sample_rows,
        random_state=random_state,
    )
    metrics = run_training(config)
    champion = metrics["champion"]["test_profiles"]["balanced_f2"]
    typer.echo(
        (
            f"Champion: {metrics['champion']['model_name']} | "
            f"AP={champion['average_precision']:.3f} | "
            f"Precision={champion['precision']:.3f} | "
            f"Recall={champion['recall']:.3f}"
        )
    )


@app.command()
def predict(
    input_csv: Path = typer.Argument(..., exists=True, help="CSV file with raw transaction features."),
    output_csv: Path = typer.Option(
        Path("artifacts/predictions_from_cli.csv"),
        help="Where to write scored predictions.",
    ),
    bundle_path: Path = typer.Option(
        Path("artifacts/model_bundle.joblib"),
        help="Trained bundle produced by the training command.",
    ),
    threshold_profile: str = typer.Option(
        "balanced_f2",
        help="Threshold profile to use when generating labels.",
    ),
) -> None:
    bundle = load_bundle(bundle_path)
    frame = pd.read_csv(input_csv)
    predictions = score_transactions(frame, bundle, threshold_profile=threshold_profile)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_csv, index=False)
    typer.echo(f"Wrote {len(predictions)} scored rows to {output_csv}.")


@app.command()
def summary(
    metrics_path: Path = typer.Option(
        Path("artifacts/metrics.json"),
        help="Metrics JSON written by the training command.",
    )
) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    typer.echo(json.dumps(metrics["champion"], indent=2))


if __name__ == "__main__":
    app()

