"""
Train and save the progression model used by the Streamlit app.

Matches: Notebooks/model w bta3 predections w elshoghl dh.ipynb
  - Data: df_clean_AD.csv
  - Model: Logistic Regression (tuned hyperparameters from notebook)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import MODEL_FEATURES, SCALE_FEATURES

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Data" / "Processed" / "df_clean_AD.csv"
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "progressor_model.joblib"
META_PATH = MODEL_DIR / "feature_meta.json"


def build_model() -> LogisticRegression:
    return LogisticRegression(
        penalty="l1",
        C=0.577493255516795,
        class_weight="balanced",
        solver="saga",
        max_iter=1000,
        random_state=42,
    )


def feature_meta_from_data(df: pd.DataFrame) -> dict:
    meta = {}
    for col in MODEL_FEATURES:
        series = df[col]
        meta[col] = {
            "min": float(series.quantile(0.01)),
            "max": float(series.quantile(0.99)),
            "default": float(series.median()),
        }
        if col in ("PTGENDER", "PTMARRY", "APOE4_COUNT"):
            meta[col]["default"] = int(round(meta[col]["default"]))
    return meta


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df[MODEL_FEATURES].copy()
    y = df["PROGRESSOR"].copy()

    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[SCALE_FEATURES] = scaler.fit_transform(X[SCALE_FEATURES])

    model = build_model()
    model.fit(X_scaled, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "scaler": scaler,
        "model_features": MODEL_FEATURES,
        "scale_features": SCALE_FEATURES,
        "model_name": "logistic_regression",
        "threshold": None,
        "dataset": "df_clean_AD.csv",
        "target": "PROGRESSOR",
        "target_labels": {0: "Non-progressor", 1: "Progressor"},
    }
    joblib.dump(bundle, MODEL_PATH)

    meta = feature_meta_from_data(df)
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved feature defaults/ranges -> {META_PATH}")
    print(f"Training samples: {len(df)} | progressors: {int(y.sum())} ({y.mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
