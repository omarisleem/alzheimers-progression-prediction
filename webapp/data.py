from __future__ import annotations

import numpy as np

from config import INTEGER_FEATURES, MODEL_FEATURES


def generate_random_features(
    feature_meta: dict,
    seed: int | None = None,
) -> dict:
    """Sample plausible feature values from training ranges (not a dataset row)."""
    rng = np.random.default_rng(seed)
    features: dict = {}

    for col in MODEL_FEATURES:
        m = feature_meta.get(col, {})
        lo = float(m.get("min", 0.0))
        hi = float(m.get("max", lo + 1.0))
        if lo >= hi:
            hi = lo + 1.0

        if col in INTEGER_FEATURES:
            allowed = m.get("values")
            if allowed:
                features[col] = int(rng.choice(allowed))
            else:
                features[col] = int(rng.integers(int(lo), int(hi) + 1))
        else:
            val = float(rng.uniform(lo, hi))
            if abs(val) >= 100 or col.endswith("VOL") or col == "BRAINVOL":
                features[col] = round(val, 1)
            elif col in {"MMSCORE", "CDRSB", "RAVLT_AVERAGE"}:
                features[col] = round(val, 2)
            else:
                features[col] = round(val, 3)

    return features
