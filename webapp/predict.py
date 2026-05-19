from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import MODEL_FEATURES, TARGET_LABELS

WEBAPP_DIR = Path(__file__).resolve().parent
MODEL_DIR = WEBAPP_DIR / "models"
REGISTRY_PATH = MODEL_DIR / "registry.json"
META_PATH = MODEL_DIR / "feature_meta.json"


def load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Model registry not found at {REGISTRY_PATH}. Run: python webapp/train_model.py"
        )
    data = json.loads(REGISTRY_PATH.read_text())
    return data["models"]


def load_feature_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {}


def _load_sklearn_bundle(model_id: str) -> dict:
    path = MODEL_DIR / f"{model_id}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model file missing: {path}")
    return joblib.load(path)


def _predict_sklearn(bundle: dict, features: dict) -> dict:
    model = bundle["model"]
    scaler = bundle["scaler"]
    scale_features = bundle["scale_features"]
    threshold = bundle.get("threshold")
    labels = bundle.get("target_labels", TARGET_LABELS)

    row = pd.DataFrame([[features[c] for c in MODEL_FEATURES]], columns=MODEL_FEATURES)
    row_scaled = row.copy()
    row_scaled[scale_features] = scaler.transform(row[scale_features])

    proba = float(model.predict_proba(row_scaled)[0, 1])
    if threshold is not None:
        pred = int(proba >= threshold)
    else:
        pred = int(model.predict(row_scaled)[0])

    return {
        "prediction": pred,
        "label": labels[pred],
        "probability": proba,
        "probability_non_progressor": 1.0 - proba,
        "model_id": bundle["model_id"],
        "model_name": bundle.get("display_name", bundle["model_id"]),
        "threshold": threshold,
    }


def _predict_keras(entry: dict, features: dict) -> dict:
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError("Neural network requires tensorflow. pip install tensorflow") from exc

    model_path = MODEL_DIR / entry["file"]
    scaler_path = MODEL_DIR / entry["scaler_file"]
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    row = pd.DataFrame([[features[c] for c in MODEL_FEATURES]], columns=MODEL_FEATURES)
    scale_features = [f for f in MODEL_FEATURES if f != "PTGENDER"]
    row_scaled = row.copy()
    row_scaled[scale_features] = scaler.transform(row[scale_features])

    X = row_scaled.values.astype(np.float32)
    proba = float(model.predict(X, verbose=0).reshape(-1)[0])
    threshold = entry.get("threshold", 0.5)
    pred = int(proba >= threshold)

    return {
        "prediction": pred,
        "label": TARGET_LABELS[pred],
        "probability": proba,
        "probability_non_progressor": 1.0 - proba,
        "model_id": entry["id"],
        "model_name": entry["name"],
        "threshold": threshold,
    }


def predict_progression(features: dict, model_id: str) -> dict:
    registry = load_registry()
    entry = next((m for m in registry if m["id"] == model_id), None)
    if entry is None:
        raise ValueError(f"Unknown model: {model_id}")

    if entry.get("backend") == "keras":
        return _predict_keras(entry, features)

    bundle = _load_sklearn_bundle(model_id)
    return _predict_sklearn(bundle, features)


def list_models() -> list[dict]:
    return load_registry()


def predict_all_models(features: dict) -> list[dict]:
    """Run prediction with every registered model; each item includes result or error."""
    outcomes = []
    for entry in load_registry():
        model_id = entry["id"]
        try:
            result = predict_progression(features, model_id)
            outcomes.append({"ok": True, **result})
        except Exception as exc:
            outcomes.append(
                {
                    "ok": False,
                    "model_id": model_id,
                    "model_name": entry.get("name", model_id),
                    "threshold": entry.get("threshold"),
                    "error": str(exc),
                }
            )
    return outcomes
