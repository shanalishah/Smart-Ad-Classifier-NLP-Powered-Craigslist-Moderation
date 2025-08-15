# streamlit_app.py — Smart Ad Classifier (reuses your 25-keyword model)
from __future__ import annotations
import streamlit as st
import pandas as pd
import joblib, pickle
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(page_title="Smart Ad Classifier – Craigslist Listings", layout="wide")
st.title("Smart Ad Classifier – Computers vs Computer Parts")
st.write(
    "Paste listings (one per line) to classify whether they describe a **Computer** or a **Computer Part**. "
    "You can also upload a CSV on the second tab."
)

# ------------------------
# Your keyword feature set (must match training)
# ------------------------
KEYWORDS: List[str] = [
    "laptop","dell","gaming","intel","core","window","pc","gb","ssd","latitude",
    "keyboard","ram","computer","mouse","mac","graphic","ryzen","monitor","cable",
    "printer","hp","router","cartridge","ink","drive"
]
N_FEATS = len(KEYWORDS)

class KeywordFeaturizer(BaseEstimator, TransformerMixin):
    """
    Build the exact 25 binary features used in your final notebook.
    For each keyword k in KEYWORDS: feature = int(k in text_lower)
    """
    def __init__(self, keywords: List[str]):
        self.keywords = list(keywords)

    def fit(self, X, y=None):
        # nothing to learn — fixed feature set
        return self

    def transform(self, X):
        # X is an iterable of raw text
        feats = []
        for t in X:
            s = str(t).lower()
            row = [1 if k in s else 0 for k in self.keywords]
            feats.append(row)
        return np.asarray(feats, dtype=np.float32)

# ------------------------
# Load your pickled LogisticRegression
# ------------------------
ROOT = Path(".")
MODEL_PATHS = [
    ROOT / "models" / "final_logistic_model.pkl",
    ROOT / "final_logistic_model.pkl",
]

def load_clf() -> Optional[object]:
    for p in MODEL_PATHS:
        if p.exists():
            try:
                # Try joblib first, then pickle
                try:
                    clf = joblib.load(p)
                except Exception:
                    with open(p, "rb") as f:
                        clf = pickle.load(f)
                return clf
            except Exception:
                continue
    return None

clf = load_clf()

# Validate model file
if clf is None:
    st.error("Model file not found. Please add `models/final_logistic_model.pkl` to the repository.")
    st.stop()

# Your LR expects 25 features (from the notebook). Verify:
n_in = getattr(clf, "n_features_in_", None)
if n_in is not None and n_in != N_FEATS:
    st.error(f"Loaded model expects {n_in} features, but the keyword featurizer provides {N_FEATS}. "
             "Please ensure the uploaded model matches the 25-keyword feature set used in training.")
    st.stop()

# Build a runtime pipeline: [KeywordFeaturizer] -> [Your LogisticRegression]
pipe = Pipeline([
    ("kw", KeywordFeaturizer(KEYWORDS)),
    ("clf", clf),
])

# ------------------------
# Helpers
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
# UI
# ------------------------
tab_try, tab_bulk = st.tabs(["Try It Now", "Bulk Classification (CSV)"])

# ---- Try It Now
with tab_try:
    st.subheader("Try It Now")
    default_examples = (
        "HP 24-inch monitor with HDMI, great condition\n"
        "Dell XPS 13 i7, 16GB RAM, 512GB SSD\n"
        "RTX 3060 graphics card, 12GB GDDR6, new in box\n"
        "Gaming PC Ryzen 5 5600X, 16GB DDR4, 1TB NVMe\n"
        "Apple MacBook Pro 16-inch, M1 Pro, 16GB RAM, 1TB SSD\n"
    )
    user_text = st.text_area(
        "Enter one listing per line:",
        height=180,
        value=default_examples,
        placeholder="e.g., Apple MacBook Air M2, 8GB RAM, 256GB SSD",
        key="ta_try",
    )
    flag_thresh = st.slider("Flag if confidence below", 0.50, 0.95, 0.65, 0.01, key="slider_try")

    if st.button("Classify", key="btn_try"):
        lines = [ln.strip() for ln in user_text.splitlines() if ln.strip()]
        if not lines:
            st.warning("Please enter at least one line of text.")
        else:
            preds = pipe.predict(lines)
            # Note: your LR is binary (0/1). If it lacks predict_proba, synthesize via decision_function.
            if hasattr(clf, "predict_proba"):
                conf = pipe.predict_proba(lines).max(axis=1)
            else:
                # Fallback: uniform confidence; or compute from decision_function if available
                conf = np.full(len(lines), 0.75, dtype=float)

            labels = np.where(preds.astype(int) == 1, "computers", "computer_parts")
            df_out = pd.DataFrame({"text": lines, "predicted_label": labels, "confidence": conf})
            df_out["confidence"] = df_out["confidence"].astype(float).round(3)
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

# ---- Bulk CSV
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

        preds = pipe.predict(df_in["text"])
        if hasattr(clf, "predict_proba"):
            conf = pipe.predict_proba(df_in["text"]).max(axis=1)
        else:
            conf = np.full(len(df_in), 0.75, dtype=float)

        labels = np.where(preds.astype(int) == 1, "computers", "computer_parts")

        df_out = df_in.copy()
        df_out["predicted_label"] = labels
        df_out["confidence"] = pd.Series(conf, index=df_in.index).round(3)

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

# ---- Footer
st.caption("Pre-trained logistic model loaded with the original keyword features.")
