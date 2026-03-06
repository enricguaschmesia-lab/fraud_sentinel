# Fraud Sentinel

Fraud Sentinel is a credit-card fraud detection system built around a reproducible training pipeline, threshold-aware evaluation, batch scoring, a Flask prediction API, and a Streamlit dashboard. It trains multiple models on the public ULB credit-card fraud dataset, selects a champion on a time-ordered validation split, and exports the model and reports needed for local use and inspection.

## Overview

The project is designed around a realistic fraud-operations workflow:

- train several candidate models on an imbalanced fraud dataset
- evaluate them on chronological validation and test splits
- convert raw scores into operating thresholds for different alert volumes
- export a reusable model bundle and supporting reports
- serve predictions through CLI, API, and dashboard interfaces

The current trained champion is a `random_forest` model.

## Current trained state

Holdout test metrics for the default `balanced_f2` threshold profile:

- Average precision: `0.810`
- ROC AUC: `0.981`
- Precision: `0.905`
- Recall: `0.760`
- F2: `0.785`
- Alerts: `63` across `56,962` test rows

Additional threshold profiles exported with the trained bundle:

- `strict_queue`: precision `1.000`, recall `0.427`, `32` alerts
- `analyst_queue`: same as the selected balanced profile for the current run
- `aggressive_queue`: precision `0.659`, recall `0.800`, `91` alerts

Top feature importances from the trained champion:

- `V14`
- `V17`
- `V12`
- `V10`
- `V16`

Largest train/test drift signals:

- `Time`
- `hour_sin`
- `hour_cos`

## Features

- Time-ordered train, validation, and test split
- Candidate model comparison across supervised and anomaly-detection baselines
- Threshold profile generation for different analyst queue sizes
- Business-aware evaluation using review-cost and recovered-fraud assumptions
- Exported feature importance and drift reports
- Batch scoring from the command line
- Local Flask API for programmatic predictions
- Streamlit dashboard for leaderboard inspection and CSV scoring
- Automated tests for the feature builder, training pipeline, and API

## Repository layout

```text
.
|-- data/
|   `-- raw/
|       `-- creditcard.csv
|-- examples/
|   |-- sample_transactions.csv
|   `-- sample_transactions_with_labels.csv
|-- src/fraud_sentinel/
|   |-- api.py
|   |-- cli.py
|   |-- config.py
|   |-- dashboard.py
|   |-- data.py
|   |-- evaluation.py
|   |-- features.py
|   |-- inference.py
|   |-- models.py
|   |-- pipeline.py
|   `-- reporting.py
|-- tests/
|   |-- conftest.py
|   |-- test_api.py
|   |-- test_features.py
|   `-- test_pipeline.py
|-- artifacts/
|   |-- metrics.json
|   |-- leaderboard.csv
|   |-- model_bundle.joblib
|   |-- test_predictions.csv
|   |-- feature_importance.csv
|   |-- drift_report.csv
|   `-- figures/
|       |-- precision_recall_curve.png
|       |-- threshold_tradeoffs.png
|       |-- feature_importance.png
|       `-- drift_report.png
|-- Dockerfile
|-- pyproject.toml
|-- pytest.ini
`-- README.md
```

## Dataset

The project expects the public fraud dataset at `data/raw/creditcard.csv`.

Download command for PowerShell:

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
curl.exe -L -C - https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv -o data\raw\creditcard.csv
```

The dataset contains anonymized PCA-style features `V1` to `V28`, plus `Time`, `Amount`, and the binary target `Class`.

## Installation

The code was exercised in this environment with `PYTHONPATH=src`. If you want a standard local install, the project is also packaged through `pyproject.toml`.

Option 1, run directly from source:

```powershell
$env:PYTHONPATH = "src"
```

Option 2, install the package:

```powershell
python -m pip install -e .
```

## Training

Run the full training pipeline:

```powershell
$env:PYTHONPATH = "src"
python -m fraud_sentinel.cli train
```

Run a faster experiment on fewer rows:

```powershell
$env:PYTHONPATH = "src"
python -m fraud_sentinel.cli train --sample-rows 50000
```

Training writes all generated outputs into `artifacts/`.

## Inference

### CLI batch scoring

Score a CSV file of raw transaction features:

```powershell
$env:PYTHONPATH = "src"
python -m fraud_sentinel.cli predict examples\sample_transactions.csv --output-csv artifacts\predictions.csv
```

The output includes:

- `risk_score`
- `predicted_label`
- `predicted_label_name`
- `threshold_profile`
- `threshold_used`
- `reason_codes`

### API

Start the Flask API:

```powershell
$env:PYTHONPATH = "src"
python -m fraud_sentinel.api
```

Available endpoints:

- `GET /health`
- `GET /metadata`
- `POST /predict`

Example prediction request:

```json
{
  "threshold_profile": "balanced_f2",
  "records": [
    {
      "Time": 0.0,
      "V1": -1.3598071336738,
      "V2": -0.0727811733098497,
      "V3": 2.53634673796914,
      "V4": 1.37815522427443,
      "V5": -0.338320769942518,
      "V6": 0.462387777762292,
      "V7": 0.239598554061257,
      "V8": 0.0986979012610507,
      "V9": 0.363786969611213,
      "V10": 0.0907941719789316,
      "V11": -0.551599533260813,
      "V12": -0.617800855762348,
      "V13": -0.991389847235408,
      "V14": -0.311169353699879,
      "V15": 1.46817697209427,
      "V16": -0.470400525259478,
      "V17": 0.207971241929242,
      "V18": 0.0257905801985591,
      "V19": 0.403992960255733,
      "V20": 0.251412098239705,
      "V21": -0.018306777944153,
      "V22": 0.277837575558899,
      "V23": -0.110473910188767,
      "V24": 0.0669280749146731,
      "V25": 0.128539358273528,
      "V26": -0.189114843888824,
      "V27": 0.133558376740387,
      "V28": -0.0210530534538215,
      "Amount": 149.62
    }
  ]
}
```

### Dashboard

Launch the Streamlit dashboard:

```powershell
$env:PYTHONPATH = "src"
python -m streamlit run src/fraud_sentinel/dashboard.py
```

The dashboard reads the generated artifacts, displays the leaderboard and threshold profiles, shows exported plots, and supports CSV upload for batch scoring.

## Modeling pipeline

1. Load the raw fraud dataset and sort by `Time`.
2. Split it chronologically into `64% train`, `16% validation`, and `20% test`.
3. Add row-local engineered features:
   - `log_amount`
   - `amount_fractional`
   - `hour_sin`
   - `hour_cos`
   - `day_index`
4. Train four candidate models:
   - logistic regression
   - random forest
   - balanced random forest
   - isolation forest
5. Build threshold profiles from validation-set scores.
6. Select the champion by the highest validation `balanced_f2`.
7. Evaluate the chosen thresholds on the holdout test set.
8. Export the model bundle, reports, tables, and figures.

## Generated artifacts

After training, the main outputs are:

- `artifacts/model_bundle.joblib`
  - serialized model, feature builder, thresholds, and feature statistics
- `artifacts/metrics.json`
  - full experiment summary and champion metrics
- `artifacts/leaderboard.csv`
  - challenger comparison table
- `artifacts/test_predictions.csv`
  - scored holdout set with labels and reason-code strings
- `artifacts/feature_importance.csv`
  - champion feature importance values
- `artifacts/drift_report.csv`
  - feature-level PSI drift report
- `artifacts/figures/precision_recall_curve.png`
- `artifacts/figures/threshold_tradeoffs.png`
- `artifacts/figures/feature_importance.png`
- `artifacts/figures/drift_report.png`

## Testing

Run the automated tests with:

```powershell
python -m pytest -q
```

The current test suite covers:

- feature generation
- artifact creation through the training pipeline
- API prediction behavior

## Docker

A `Dockerfile` is included for containerized execution. The current local workflow in this repository uses the source-based commands shown above.

## Notes

- The dataset is highly imbalanced, so threshold selection matters as much as raw ranking performance.
- Because the public benchmark anonymizes most inputs, the project emphasizes modeling, thresholding, and deployment workflow more than domain-specific feature interpretation.
- All generated artifacts, raw data, and local notes are excluded by `.gitignore`.
