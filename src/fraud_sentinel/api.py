from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

from fraud_sentinel.inference import load_bundle, score_transactions
from fraud_sentinel.pipeline import default_bundle_path


def create_app(bundle_path: Path | None = None) -> Flask:
    app = Flask(__name__)
    resolved_bundle_path = bundle_path or default_bundle_path()

    @app.get("/health")
    def health() -> tuple[dict[str, object], int]:
        bundle_exists = resolved_bundle_path.exists()
        return {
            "status": "ok" if bundle_exists else "missing_bundle",
            "bundle_path": str(resolved_bundle_path),
        }, 200 if bundle_exists else 503

    @app.get("/metadata")
    def metadata() -> tuple[dict[str, object], int]:
        bundle = load_bundle(resolved_bundle_path)
        return {
            "model_name": bundle["model_name"],
            "threshold_profiles": bundle["thresholds"],
            "metadata": bundle["metadata"],
        }, 200

    @app.post("/predict")
    def predict() -> tuple[dict[str, object], int]:
        if not request.is_json:
            return {"error": "Request body must be JSON."}, 400

        payload = request.get_json(silent=True) or {}
        records = payload.get("records")
        threshold_profile = payload.get("threshold_profile", "balanced_f2")
        if not isinstance(records, list) or not records:
            return {"error": "Provide a non-empty 'records' array."}, 400

        bundle = load_bundle(resolved_bundle_path)
        frame = pd.DataFrame.from_records(records)
        predictions = score_transactions(frame, bundle, threshold_profile=threshold_profile)
        serializable = json.loads(predictions.to_json(orient="records"))
        return {
            "model_name": bundle["model_name"],
            "threshold_profile": threshold_profile,
            "predictions": serializable,
        }, 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

