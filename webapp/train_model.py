"""
Train and save all progression models from the modeling notebook.

Notebook: model w bta3 predections w elshoghl dh.ipynb
Data: df_clean_AD.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import INTEGER_FEATURES, MODEL_FEATURES, SCALE_FEATURES, TARGET_LABELS

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "Data" / "Processed" / "df_clean_AD.csv"
MODEL_DIR = Path(__file__).resolve().parent / "models"
REGISTRY_PATH = MODEL_DIR / "registry.json"
META_PATH = MODEL_DIR / "feature_meta.json"


def feature_meta_from_data(df: pd.DataFrame) -> dict:
    meta = {}
    for col in MODEL_FEATURES:
        series = df[col].dropna()
        entry = {
            "min": float(series.quantile(0.01)),
            "max": float(series.quantile(0.99)),
            "default": float(series.median()),
        }
        if col in INTEGER_FEATURES:
            entry["default"] = int(round(entry["default"]))
            entry["values"] = sorted(int(v) for v in series.unique())
        meta[col] = entry
    return meta


def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X[SCALE_FEATURES])
    return scaler


def scale_frame(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    out = X.copy()
    out[SCALE_FEATURES] = scaler.transform(X[SCALE_FEATURES])
    return out


def save_sklearn_bundle(
    model_id: str,
    display_name: str,
    model,
    scaler: StandardScaler,
    threshold: float | None,
) -> dict:
    path = MODEL_DIR / f"{model_id}.joblib"
    bundle = {
        "model": model,
        "scaler": scaler,
        "model_features": MODEL_FEATURES,
        "scale_features": SCALE_FEATURES,
        "model_id": model_id,
        "display_name": display_name,
        "threshold": threshold,
        "backend": "sklearn",
        "target_labels": TARGET_LABELS,
    }
    joblib.dump(bundle, path)
    return {
        "id": model_id,
        "name": display_name,
        "file": path.name,
        "threshold": threshold,
        "backend": "sklearn",
    }


def train_logistic_regression(X: pd.DataFrame, y: pd.Series) -> tuple:
    scaler = fit_scaler(X)
    Xs = scale_frame(X, scaler)
    model = LogisticRegression(
        penalty="l1",
        C=0.577493255516795,
        class_weight="balanced",
        solver="saga",
        max_iter=1000,
        random_state=42,
    )
    model.fit(Xs, y)
    return model, scaler, None


def train_svm(X: pd.DataFrame, y: pd.Series) -> tuple:
    scaler = fit_scaler(X)
    Xs = scale_frame(X, scaler)
    model = SVC(
        kernel="linear",
        C=3,
        class_weight="balanced",
        probability=True,
        random_state=42,
    )
    model.fit(Xs, y)
    return model, scaler, 0.25


def train_random_forest(X: pd.DataFrame, y: pd.Series) -> tuple:
    scaler = fit_scaler(X)
    Xs = scale_frame(X, scaler)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        min_samples_split=20,
        max_features="sqrt",
        max_samples=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(Xs, y)
    return model, scaler, 0.3


def train_xgboost(X: pd.DataFrame, y: pd.Series) -> tuple:
    import xgboost as xgb

    scaler = fit_scaler(X)
    Xs = scale_frame(X, scaler)
    model = xgb.XGBClassifier(
        booster="gblinear",
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=11.249850857719316,
        reg_alpha=0.004011413377415873,
        reg_lambda=0.022525794833067405,
        learning_rate=0.16907636777548096,
        n_estimators=529,
        random_state=42,
        verbosity=0,
        updater="coord_descent",
    )
    model.fit(Xs, y)
    return model, scaler, 0.55


def train_neural_network(X: pd.DataFrame, y: pd.Series) -> dict | None:
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except ImportError:
        print("  [skip] Neural network — tensorflow not installed")
        return None

    scaler = fit_scaler(X)
    Xs = scale_frame(X, scaler).values.astype(np.float32)
    y_arr = y.values.astype(np.float32)

    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    class_weight = {0: 1.0, 1: neg / pos * 1.3}

    tf.random.set_seed(42)

    inp = keras.Input(shape=(Xs.shape[1],))
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        EarlyStopping(
            monitor="loss",
            patience=20,
            mode="min",
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=10,
            mode="min",
            min_lr=1e-6,
            verbose=0,
        ),
    ]
    model.fit(
        Xs,
        y_arr,
        epochs=300,
        batch_size=8,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )

    model_id = "neural_network"
    keras_path = MODEL_DIR / f"{model_id}.keras"
    scaler_path = MODEL_DIR / f"{model_id}_scaler.joblib"
    model.save(keras_path)
    joblib.dump(scaler, scaler_path)

    return {
        "id": model_id,
        "name": "Neural Network (Keras)",
        "file": keras_path.name,
        "scaler_file": scaler_path.name,
        "threshold": 0.5,
        "backend": "keras",
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df[MODEL_FEATURES].copy()
    y = df["PROGRESSOR"].copy()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    registry: list[dict] = []

    trainers = [
        ("logistic_regression", "Logistic Regression", train_logistic_regression),
        ("svm", "SVM (linear)", train_svm),
        ("random_forest", "Random Forest", train_random_forest),
    ]

    for model_id, display_name, train_fn in trainers:
        print(f"Training {display_name}...")
        model, scaler, threshold = train_fn(X, y)
        entry = save_sklearn_bundle(model_id, display_name, model, scaler, threshold)
        registry.append(entry)
        print(f"  -> {MODEL_DIR / entry['file']}")

    try:
        print("Training XGBoost...")
        model, scaler, threshold = train_xgboost(X, y)
        entry = save_sklearn_bundle("xgboost", "XGBoost (gblinear)", model, scaler, threshold)
        registry.append(entry)
        print(f"  -> {MODEL_DIR / entry['file']}")
    except ImportError:
        print("  [skip] XGBoost — pip install xgboost")

    nn_entry = train_neural_network(X, y)
    if nn_entry:
        registry.append(nn_entry)
        print(f"  -> {MODEL_DIR / nn_entry['file']}")

    REGISTRY_PATH.write_text(json.dumps({"models": registry}, indent=2))
    META_PATH.write_text(json.dumps(feature_meta_from_data(df), indent=2))

    print(f"\nSaved {len(registry)} models -> {REGISTRY_PATH}")
    print(f"Feature meta -> {META_PATH}")
    print(f"Samples: {len(df)} | progressors: {int(y.sum())} ({y.mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
