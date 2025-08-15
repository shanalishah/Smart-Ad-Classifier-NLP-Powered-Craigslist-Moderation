# streamlit_app.py — Smart Ad Classifier (single unified pipeline)
from __future__ import annotations
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

# ------------------------
# Page configuration
# ------------------------
st.set_page_config(page_title="Smart Ad Classifier – Craigslist Listings", layout="wide")
st.title("Smart Ad Classifier – Computers vs Computer Parts")
st.write(
    "Paste listings (one per line) to classify whether they describe a **Computer** or a **Computer Part**. "
    "You can also upload a CSV in the second tab."
)

# ------------------------
# Model locations (prefer root, then /models)
# ------------------------
ROOT = Path(".")
PIPELINE_PATHS = [
    ROOT / "pipeline_lr_tfidf.joblib",
    ROOT / "models" / "pipeline_lr_tfidf.joblib",
]

@st.cache_resource
def load_pipeline() -> Optional[object]:
    for p in PIPELINE_PATHS:
        if p.exists():
            try:
                pipe = joblib.load(p)
                # quick smoke test
                _ = pipe.predict(["sample text"])
                return pipe
            except Exception:
                continue
    return None

pipe = load_pipeline()

# ------------------------
# Helper: normalize CSV input
# ------------------------
def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if "text" in cols:
        text = df[cols["text"]].fillna("").astype(str)
    elif {"title", "description"}.issubset(set(cols)):
        text = (df[cols["title"]].fillna("") + " " + df[cols["description"]].fillna("")).astype(str)
    else:
        st.error("Your file must contain a 'text' column OR both 'title' and 'description'.")
        st.stop()
    out = pd.DataFrame({"text": text})
    out = out[out["text"].str.strip().ne("")].reset_index(drop=True)
    return out

# ------------------------
# Tabs
# ------------------------
tab_try, tab_bulk = st.tabs(["Try It Now", "Bulk Classification (CSV)"])

# ---- Tab: Try It Now
with tab_try:
    st.subheader("Try It Now")
    default_examples = (
        "HP 24-inch monitor with HDMI, great condition\n"
        "Dell XPS 13 i7, 16GB RAM, 512GB SSD\n"
        "RTX 3060 graphics card, 12GB GDDR6, new in box\n"
        "Gaming PC Ryzen 5 5600X, 16GB DDR4, 1TB NVMe\n"
    )
    user_text = st.text_area(
        "Enter one listing per line:",
        height=160,
        value=default_examples,
        placeholder="e.g., RTX 3060 graphics card, 12GB GDDR6, new in box",
        key="ta_try",
    )
    flag_thresh = st.slider("Flag if confidence below", 0.50, 0.95, 0.65, 0.01, key="slider_try")

    if st.button("Classify", key="btn_try"):
        lines = [ln.strip() for ln in user_text.splitlines() if ln.strip()]
        if not lines:
            st.warning("Please enter at least one line of text.")
        elif pipe is None:
            st.error("Model file not found. Please add `pipeline_lr_tfidf.joblib` to the repository.")
        else:
            preds = pipe.predict(lines)
            conf = pipe.predict_proba(lines).max(axis=1)
            df_out = pd.DataFrame({"text": lines, "predicted_label": preds, "confidence": conf})
            df_out["confidence"] = df_out["confidence"].round(3)
            df_out["flag_low_conf"] = df_out["confidence"] < flag_thresh

            st.subheader("Results")
            st.dataframe(df_out, use_container_width=True)

            csv_try = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions",
                data=csv_try,
                file_name="predictions.csv",
                mime="text/csv",
                key="dl_predictions_try",
            )

# ---- Tab: Bulk CSV
with tab_bulk:
    st.subheader("Bulk Classification (CSV)")
    uploaded = st.file_uploader(
        "Upload a CSV",
        type=["csv"],
        help="Use a single 'text' column or both 'title' and 'description'.",
        key="csv_upload",
    )

    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        df_in = normalize_input(df_raw)

        if pipe is None:
            st.error("Model file not found. Please add `pipeline_lr_tfidf.joblib` to the repository.")
        else:
            preds = pipe.predict(df_in["text"])
            conf = pipe.predict_proba(df_in["text"]).max(axis=1)
            df_out = df_in.copy()
            df_out["predicted_label"] = preds
            df_out["confidence"] = pd.Series(conf).round(3)

            st.dataframe(df_out, use_container_width=True)

            csv_bulk = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions",
                data=csv_bulk,
                file_name="predictions.csv",
                mime="text/csv",
                key="dl_predictions_bulk",
            )
    else:
        st.info("No file uploaded yet.")

# ---- Footer (polite, non-technical)
if pipe is not None:
    st.caption("Pre-trained model loaded for instant results.")
else:
    st.caption("Add the trained model file `pipeline_lr_tfidf.joblib` to enable predictions.")
