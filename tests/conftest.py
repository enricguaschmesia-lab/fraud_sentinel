from __future__ import annotations

from pathlib import Path
import shutil
import sys
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture()
def workspace_tmp_dir() -> Path:
    base_dir = PROJECT_ROOT / "outputs" / "test_runs"
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / f"run_{uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield run_dir
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


@pytest.fixture()
def synthetic_dataset_path(workspace_tmp_dir: Path) -> Path:
    X, y = make_classification(
        n_samples=3_200,
        n_features=28,
        n_informative=12,
        n_redundant=6,
        n_clusters_per_class=2,
        weights=[0.97, 0.03],
        flip_y=0.002,
        class_sep=1.2,
        random_state=42,
    )

    frame = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    frame.insert(0, "Time", np.arange(len(frame)) * 13.0)
    frame["Amount"] = np.exp(np.random.default_rng(42).normal(loc=3.8, scale=0.6, size=len(frame)))
    frame["Class"] = y.astype(int)

    csv_path = workspace_tmp_dir / "synthetic_creditcard.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def raw_feature_frame(synthetic_dataset_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(synthetic_dataset_path)
    return frame.drop(columns=["Class"])
