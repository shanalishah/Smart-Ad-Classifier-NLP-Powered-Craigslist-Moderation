# streamlit_app.py — Smart Ad Classifier (uses your 25-keyword LR model)
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

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
# Your original keyword set (25 features)
# pulled from your notebook’s STEP 4
# ------------------------
KEYWORDS: List[str] = [
    "laptop","dell","gaming","intel","core","window","pc",
    "gb","ssd","latitude","keyboard","ram","computer","mouse",
    "mac","graphic","ryzen","monitor","cable","printer","hp","router",
    "cartridge","ink","drive",
]
KW_COLS = [f"kw_{kw}" for kw in KEYWORDS]

# ------------------------
# Paths for your saved model
# ------------------------
ROOT = Path(".")
MODEL_PATHS = [
    ROOT / "models" / "final_logistic_model.pkl",   # preferred
    ROOT / "data" / "final_logistic_model.pkl",     # fallback
    ROOT / "final_logistic_model.pkl",              # last resort
]

@st.cache_resource
def load_lr_model():
    for p in MODEL_PATHS:
        if p.exists():
            mdl = joblib.load(p)
            # sanity: expect 25 input features and binary classes [0,1]
            n_feat = getattr(mdl, "n_features_in_", None)
            classes = getattr(mdl, "classes_", None)
            if n_feat == len(KEYWORDS) and classes is not None and set(list(classes)) == {0,1}:
                return mdl, str(p)
    return None, None

model, model_path = load_lr_model()

# ------------------------
# Preprocessing & featureizer (exactly as in your notebook)
# ------------------------
def simple_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

def kw_features(texts: List[str]) -> np.ndarray:
    """
    Build the SAME 25-dim keyword presence features you trained the model on.
    """
    feats = np.zeros((len(texts), len(KEYWORDS)), dtype=np.int8)
    for i, t in enumerate(texts):
        s = simple_clean(t)
        for j, kw in enumerate(KEYWORDS):
            feats[i, j] = 1 if kw in s else 0
    return feats

def normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if "text" in cols:
        text = df[cols["text"]].fillna("").astype(str)
    elif {"title","description"}.issubset(set(cols)):
        text = (df[cols["title"]].fillna("") + " " + df[cols["description"]].fillna("")).astype(str)
    else:
        st.error("Your file must contain a 'text' column OR both 'title' and 'description'.")
        st.stop()
    out = pd.DataFrame({"text": text})
    out = out[out["text"].str.strip().ne("")].reset_index(drop=True)
    return out

def decode_label(yhat: int) -> str:
    # your mapping in notebook: computers -> 1, computer_parts -> 0
    return "computers" if yhat == 1 else "computer_parts"

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
        "Apple MacBook Pro 16-inch, M1, 16GB RAM\n"
        "RTX 3060 graphics card, 12GB GDDR6, new in box\n"
        "Gaming PC Ryzen 5 5600X, 16GB DDR4, 1TB NVMe\n"
    )
    user_text = st.text_area(
        "Enter one listing per line:",
        height=160,
        value=default_examples,
        placeholder="e.g., Apple MacBook Air M2, 8GB RAM, 256GB SSD",
        key="ta_try",
    )
    flag_thresh = st.slider("Flag if confidence below", 0.50, 0.95, 0.65, 0.01, key="slider_try")

    if st.button("Classify", key="btn_try"):
        lines = [ln.strip() for ln in user_text.splitlines() if ln.strip()]
        if not lines:
            st.warning("Please enter at least one line of text.")
        elif model is None:
            st.error("Model file not found or incompatible. Please add `final_logistic_model.pkl` to /models.")
        else:
            X = kw_features(lines)
            preds_raw = model.predict(X)
            probs = model.predict_proba(X).max(axis=1)

            labels = [decode_label(int(y)) for y in preds_raw]
            df_out = pd.DataFrame({
                "text": lines,
                "predicted_label": labels,
                "confidence": np.round(probs, 3),
            })
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
        df_in = normalize_input_df(df_raw)

        if model is None:
            st.error("Model file not found or incompatible. Please add `final_logistic_model.pkl` to /models.")
        else:
            X = kw_features(df_in["text"].tolist())
            preds_raw = model.predict(X)
            probs = model.predict_proba(X).max(axis=1)

            labels = [decode_label(int(y)) for y in preds_raw]
            df_out = df_in.copy()
            df_out["predicted_label"] = labels
            df_out["confidence"] = np.round(probs, 3)

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
if model_path:
    st.caption(f"Pre-trained model loaded from {model_path}.")
else:
    st.caption("Add the trained model file `final_logistic_model.pkl` to /models to enable predictions.")
