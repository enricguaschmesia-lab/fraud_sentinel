"""Fraud Sentinel package."""

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from fraud_sentinel.api import create_app
from fraud_sentinel.config import TrainingConfig
from fraud_sentinel.pipeline import run_training

__all__ = ["TrainingConfig", "create_app", "run_training"]
