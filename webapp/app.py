"""
Alzheimer progression predictor — simple Streamlit UI.

Run from project root:
  streamlit run webapp/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

WEBAPP_DIR = Path(__file__).resolve().parent
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

from config import FEATURE_UI, MODEL_FEATURES
from predict import load_feature_meta, predict_progression

st.set_page_config(
    page_title="Alzheimer Progression Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 820px; }
    .result-card {
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-top: 1rem;
        border: 1px solid #e2e8f0;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }
    .result-progressor {
        border-color: #fecaca;
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
    }
    .result-non {
        border-color: #bbf7d0;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }
    .result-title { font-size: 1.35rem; font-weight: 700; margin: 0 0 0.35rem 0; }
    .muted { color: #64748b; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _feature_meta():
    return load_feature_meta()


def _defaults():
    meta = _feature_meta()
    values = {}
    for col in MODEL_FEATURES:
        if col in meta:
            values[col] = meta[col]["default"]
        elif col in FEATURE_UI and FEATURE_UI[col].get("kind") == "select":
            values[col] = list(FEATURE_UI[col]["options"].keys())[0]
        else:
            values[col] = 0.0
    return values


def _render_inputs(defaults: dict) -> dict:
    meta = _feature_meta()
    values = {}
    cols = st.columns(2)

    for i, col in enumerate(MODEL_FEATURES):
        ui = FEATURE_UI[col]
        m = meta.get(col, {})
        target = cols[i % 2]

        with target:
            if ui["kind"] == "select":
                options = ui["options"]
                keys = list(options.keys())
                default_key = int(defaults[col]) if defaults[col] in keys else keys[0]
                idx = keys.index(default_key) if default_key in keys else 0
                choice = st.selectbox(
                    ui["label"],
                    options=keys,
                    index=idx,
                    format_func=lambda k, opts=options: opts[k],
                    help=ui.get("help"),
                    key=f"inp_{col}",
                )
                values[col] = int(choice)
            else:
                lo = float(m.get("min", defaults[col] - 10))
                hi = float(m.get("max", defaults[col] + 10))
                if lo >= hi:
                    hi = lo + 1.0
                val = st.number_input(
                    ui["label"],
                    min_value=lo,
                    max_value=hi,
                    value=float(defaults[col]),
                    help=ui.get("help"),
                    key=f"inp_{col}",
                )
                values[col] = float(val)
    return values


def main():
    with st.sidebar:
        st.title("About")
        st.markdown(
            """
            Predicts whether a patient is likely to **progress to Alzheimer's disease (AD)**
            based on baseline clinical and imaging features.

            **Model:** Logistic regression (same hyperparameters as your modeling notebook).

            **Data:** `df_clean_AD.csv`

            This tool is for **research / coursework** only — not for clinical use.
            """
        )
        if st.button("Reset form to typical values", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("inp_"):
                    del st.session_state[key]
            st.rerun()

    st.title("Alzheimer Progression Predictor")
    st.caption("Enter patient baseline features → get progressor / non-progressor prediction")

    defaults = _defaults()

    with st.form("patient_form", clear_on_submit=False):
        st.subheader("Patient features")
        features = _render_inputs(defaults)
        submitted = st.form_submit_button("Predict progression", type="primary", use_container_width=True)

    if submitted:
        try:
            result = predict_progression(features)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info("From the project root, run: `python webapp/train_model.py`")
            return

        is_prog = result["prediction"] == 1
        card_class = "result-progressor" if is_prog else "result-non"
        emoji = "⚠️" if is_prog else "✓"
        title = result["label"]
        prob = result["probability"] if is_prog else result["probability_non_progressor"]

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <p class="result-title">{emoji} {title}</p>
                <p class="muted">Model confidence: <strong>{prob * 100:.1f}%</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(prob, text=f"Progression probability: {result['probability'] * 100:.1f}%")

        c1, c2 = st.columns(2)
        c1.metric("P(progressor)", f"{result['probability'] * 100:.1f}%")
        c2.metric("P(non-progressor)", f"{result['probability_non_progressor'] * 100:.1f}%")

        with st.expander("What do these labels mean?"):
            st.markdown(
                """
                - **Progressor:** patient is predicted to reach AD (diagnosis code 3) during follow-up.
                - **Non-progressor:** patient is predicted not to progress to AD.

                The model was trained on ADNI-style cohort data with class imbalance; probabilities
                should be interpreted cautiously.
                """
            )


if __name__ == "__main__":
    main()
