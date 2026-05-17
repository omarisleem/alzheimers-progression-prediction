from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import MODEL_FEATURES

WEBAPP_DIR = Path(__file__).resolve().parent
MODEL_PATH = WEBAPP_DIR / "models" / "progressor_model.joblib"
META_PATH = WEBAPP_DIR / "models" / "feature_meta.json"


def load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python webapp/train_model.py"
        )
    return joblib.load(MODEL_PATH)


def load_feature_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}


def predict_progression(features: dict) -> dict:
    bundle = load_bundle()
    model = bundle["model"]
    scaler = bundle["scaler"]
    scale_features = bundle["scale_features"]
    labels = bundle["target_labels"]

    row = pd.DataFrame([[features[c] for c in MODEL_FEATURES]], columns=MODEL_FEATURES)
    row_scaled = row.copy()
    row_scaled[scale_features] = scaler.transform(row[scale_features])

    proba = float(model.predict_proba(row_scaled)[0, 1])
    pred = int(model.predict(row_scaled)[0])
    return {
        "prediction": pred,
        "label": labels[pred],
        "probability": proba,
        "probability_non_progressor": 1.0 - proba,
    }
