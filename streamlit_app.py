# streamlit_app.py — loads unified pipeline_lr_tfidf.joblib
from __future__ import annotations
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Smart Ad Classifier – Craigslist Listings", layout="wide")
st.title("Smart Ad Classifier – Computers vs Computer Parts")
st.write("Paste listings (one per line) or upload a CSV to classify them.")

ROOT = Path(".")
PIPELINE_PATHS = [
    ROOT / "pipeline_lr_tfidf.joblib",
    ROOT / "models" / "pipeline_lr_tfidf.joblib",
]

@st.cache_resource
def load_pipeline():
    for p in PIPELINE_PATHS:
        if p.exists():
            pipe = joblib.load(p)
            _ = pipe.predict(["sanity check"])
            return pipe
    return None

pipe = load_pipeline()

def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if "text" in cols:
        text = df[cols["text"]].fillna("").astype(str)
    elif {"title","description"}.issubset(set(cols)):
        text = (df[cols["title"]].fillna("") + " " + df[cols["description"]].fillna("")).astype(str)
    else:
        st.error("CSV must contain 'text' OR both 'title' and 'description'.")
        st.stop()
    out = pd.DataFrame({"text": text})
    out = out[out["text"].str.strip().ne("")].reset_index(drop=True)
    return out

tab_try, tab_bulk = st.tabs(["Try It Now", "Bulk Classification (CSV)"])

with tab_try:
    st.subheader("Try It Now")
    default_examples = (
        "Apple MacBook Pro 14-inch, M2, 16GB RAM, 512GB SSD\n"
        "RTX 3060 graphics card, 12GB, brand new\n"
        "Lenovo ThinkPad T14 laptop, 32GB RAM, 1TB SSD\n"
        "Corsair 750W PSU power supply\n"
    )
    user_text = st.text_area("Enter one listing per line:", height=180, value=default_examples)
    flag_thresh = st.slider("Flag if confidence below", 0.50, 0.95, 0.65, 0.01)

    if st.button("Classify"):
        if pipe is None:
            st.error("Model not found. Run train_pipeline.py first.")
        else:
            lines = [ln.strip() for ln in user_text.splitlines() if ln.strip()]
            preds = pipe.predict(lines)
            conf = pipe.predict_proba(lines).max(axis=1)
            df_out = pd.DataFrame({"text": lines, "predicted_label": preds, "confidence": conf.round(3)})
            df_out["flag_low_conf"] = df_out["confidence"] < flag_thresh
            st.dataframe(df_out, use_container_width=True)
            st.download_button("Download Predictions", df_out.to_csv(index=False), "predictions.csv", key="dl_predictions_try")

with tab_bulk:
    st.subheader("Bulk Classification (CSV)")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"], help="Use a 'text' column or both 'title' and 'description'.")
    if uploaded is not None:
        if pipe is None:
            st.error("Model not found. Run train_pipeline.py first.")
        else:
            df_raw = pd.read_csv(uploaded)
            df_in = normalize_input(df_raw)
            preds = pipe.predict(df_in["text"])
            conf = pipe.predict_proba(df_in["text"]).max(axis=1)
            df_out = df_in.copy()
            df_out["predicted_label"] = preds
            df_out["confidence"] = conf.round(3)
            st.dataframe(df_out, use_container_width=True)
            st.download_button("Download Predictions", df_out.to_csv(index=False), "predictions.csv", key="dl_predictions_bulk")
    else:
        st.info("No file uploaded yet.")

st.caption("Pre-trained TF–IDF + Logistic Regression pipeline loaded for instant results.")
