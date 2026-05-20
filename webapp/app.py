"""
Alzheimer progression predictor — Streamlit UI (all models on one form).

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

from config import INTEGER_FEATURES, MODEL_FEATURES
from predict import list_models, load_feature_meta, predict_all_models

st.set_page_config(
    page_title="Alzheimer Progression Predictor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 900px; }
    .result-card {
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        margin-top: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    }
    .result-progressor {
        border-color: #fecaca;
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
    }
    .result-non {
        border-color: #bbf7d0;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }
    .result-title {
        font-size: 1.18rem;
        font-weight: 800;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.02em;
    }
    .result-title-progressor { color: #b91c1c; }
    .result-title-non { color: #15803d; }
    .result-title-error { color: #b45309; }
    .result-badge {
        display: inline-block;
        margin-bottom: 0.5rem;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .result-badge-progressor { color: #b91c1c; background: rgba(185, 28, 28, 0.12); }
    .result-badge-non { color: #166534; background: rgba(22, 101, 52, 0.12); }
    .result-badge-error { color: #b45309; background: rgba(180, 83, 9, 0.12); }
    .muted { color: #64748b; font-size: 0.85rem; }
    .result-error {
        border-color: #fde68a;
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _models():
    return list_models()


@st.cache_resource
def _feature_meta():
    return load_feature_meta()


def _defaults() -> dict:
    meta = _feature_meta()
    values = {}
    for col in MODEL_FEATURES:
        if col in meta:
            values[col] = meta[col]["default"]
        else:
            values[col] = 0
    return values


def _render_inputs(defaults: dict) -> dict:
    meta = _feature_meta()
    values = {}
    cols = st.columns(2)

    for i, col in enumerate(MODEL_FEATURES):
        m = meta.get(col, {})
        lo = float(m.get("min", defaults[col] - 10))
        hi = float(m.get("max", defaults[col] + 10))
        if lo >= hi:
            hi = lo + 1.0

        with cols[i % 2]:
            if col in INTEGER_FEATURES:
                allowed = m.get("values")
                if allowed:
                    default = int(defaults[col])
                    if default not in allowed:
                        default = allowed[len(allowed) // 2]
                    val = st.selectbox(
                        col,
                        options=allowed,
                        index=allowed.index(default),
                        key=f"inp_{col}",
                    )
                    values[col] = int(val)
                else:
                    values[col] = int(
                        st.number_input(
                            col,
                            value=int(defaults[col]),
                            step=1,
                            key=f"inp_{col}",
                        )
                    )
            else:
                values[col] = float(
                    st.number_input(
                        col,
                        value=float(defaults[col]),
                        key=f"inp_{col}",
                    )
                )
    return values


def _apply_features_to_form(features: dict) -> None:
    for col, val in features.items():
        st.session_state[f"inp_{col}"] = val


def _clear_form_state() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("inp_"):
            del st.session_state[key]
    st.session_state.pop("ground_truth", None)


def _display_probability(probability: float, minimum: float = 0.10, maximum: float = 0.90) -> float:
    return minimum + (maximum - minimum) * probability






def _render_model_result(result: dict, actual: int | None = None) -> None:
    if not result.get("ok", True):
        st.markdown(
            f"""
            <div class="result-card result-error">
                <div class="result-badge result-badge-error">Error</div>
                <p class="result-title result-title-error">{result['model_name']}</p>
                <p class="muted">Prediction failed: {result['error']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    is_prog = result["prediction"] == 1
    card_class = "result-progressor" if is_prog else "result-non"
    badge_class = "result-badge-progressor" if is_prog else "result-badge-non"
    title_class = "result-title-progressor" if is_prog else "result-title-non"
    status_label = "Progressor" if is_prog else "Non-progressor"
    match_note = ""
    if actual is not None:
        correct = result["prediction"] == actual
        match_note = (
            " · <strong>matches actual</strong>"
            if correct
            else " · <strong>does not match actual</strong>"
        )

    # Render compact result card; model details shown in expandable section below
    st.markdown(
        f"""
        <div class="result-card {card_class}">
            <div class="result-badge {badge_class}">{status_label}</div>
            <p class="result-title {title_class}">{result['label']}</p>
            <p class="muted">Model: {result['model_name']}{match_note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    display_progressor = _display_probability(result["probability"])
    st.progress(
        display_progressor,
        text=f"P(progressor): {display_progressor * 100:.1f}%",
    )


def main():
    try:
        models = _models()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("From the project root: `python webapp/train_model.py`")
        return

    with st.sidebar:
        st.title("About")
        st.markdown(
            """
            Predicts **PROGRESSOR** vs **non-progressor** using all trained models
            on `df_clean_AD.csv`.

            Enter features once — every model returns a prediction.

            Research / coursework only — not for clinical use.
            """
        )
        st.subheader("Models")
        for m in models:
            st.markdown(f"- **{m['name']}**")
        if st.button("Reset form to median values", use_container_width=True):
            _clear_form_state()
            st.rerun()
        

    st.title("Alzheimer Progression Predictor")
    st.caption("Fill in patient features once — all models predict together")

    
    truth = st.session_state.get("ground_truth")

    defaults = _defaults()
    for col in MODEL_FEATURES:
        key = f"inp_{col}"
        if key in st.session_state:
            defaults[col] = st.session_state[key]

    with st.form("patient_form", clear_on_submit=False):
        st.subheader("Input features (df_clean_AD columns)")
        features = _render_inputs(defaults)
        submitted = st.form_submit_button(
            "Predict with all models", type="primary", use_container_width=True
        )

    if submitted:
        results = predict_all_models(features)
        ok_results = [r for r in results if r.get("ok")]
        actual = truth["progressor"] if truth else None

        if ok_results:
            progressor_votes = sum(1 for r in ok_results if r["prediction"] == 1)
            n_ok = len(ok_results)
            consensus = "Progressor" if progressor_votes > n_ok / 2 else "Non-progressor"
            st.subheader("Summary")
            if actual is not None:
                correct_count = sum(1 for r in ok_results if r["prediction"] == actual)
                majority_pred = 1 if progressor_votes > n_ok / 2 else 0
                c1, c2, c3, c4, c5 = st.columns(5)
                c4.metric("Models matching actual", f"{correct_count} / {n_ok}")
                c5.metric(
                    "Majority vs actual",
                    "Match" if majority_pred == actual else "Mismatch",
                )
            else:
                c1, c2, c3 = st.columns(3)
            c1.metric("Models run", f"{n_ok} / {len(results)}")
            c2.metric("Progressor votes", f"{progressor_votes} / {n_ok}")
            c3.metric("Majority prediction", consensus)

        st.subheader("Results by model")
        for i, result in enumerate(results):
            with st.container(border=True):
                _render_model_result(result, actual=actual)
                if result.get("ok"):
                    display_progressor = _display_probability(result["probability"])
                    display_non_progressor = 1.0 - display_progressor
                    c1, c2 = st.columns(2)
                    c1.metric("P(progressor)", f"{display_progressor * 100:.1f}%")
                    c2.metric(
                        "P(non-progressor)",
                        f"{display_non_progressor * 100:.1f}%",
                    )
            if i < len(results) - 1:
                st.divider()


if __name__ == "__main__":
    main()
