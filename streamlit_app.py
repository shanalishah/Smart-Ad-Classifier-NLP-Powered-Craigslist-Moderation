# streamlit_app.py — Smart Ad Classifier (professional UI; uses your 25-keyword LR model)
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# ---------------- Page setup ----------------
st.set_page_config(page_title="Smart Ad Classifier", layout="wide")
st.title("Smart Ad Classifier – Computers vs Computer Parts")
st.write(
    "Paste a listing (or several, one per line) to classify whether it describes a "
    "**Computer** or a **Computer Part**. You can also upload a CSV on the second tab."
)

# ---------------- Paths ----------------
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "final_logistic_model.pkl"   # your saved LR model (trained on 25 keywords)

# ---------------- Keyword feature maker (ORDER must match training!) ----------------
KEYWORDS = [
    "laptop","dell","gaming","intel","core","window","pc","gb","ssd","latitude",
    "keyboard","ram","computer","mouse","mac","graphic","ryzen","monitor","cable",
    "printer","hp","router","cartridge","ink","drive"
]
N_FEATS = len(KEYWORDS)

class KeywordFeaturizer(BaseEstimator, TransformerMixin):
    """Recreate the exact 25 binary features your LR was trained on."""
    def __init__(self, keywords):
        self.keywords = list(keywords)
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        feats = []
        for t in X:
            s = str(t).lower()
            feats.append([1 if k in s else 0 for k in self.keywords])
        return np.asarray(feats, dtype=np.float32)

# ---------------- Helpers ----------------
def _normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    """Accept 'text' OR ('title' + 'description') and return a clean DataFrame with 'text'."""
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

def _fallback_stub(texts: list[str]) -> Tuple[list[str], list[float]]:
    """
    Minimal heuristic so the demo still runs if the model file is missing.
    (Used only as a graceful fallback; your real model is used when present.)
    """
    parts_kw = {"gpu","graphics card","monitor","keyboard","mouse","ssd","hdd","ram","memory",
                "cpu","processor","motherboard","psu","power supply","cable","adapter","webcam"}
    comps_kw = {"laptop","notebook","macbook","desktop","tower","gaming pc","prebuilt","computer"}
    preds, probs = [], []
    for t in texts:
        s = t.lower()
        is_part = any(k in s for k in parts_kw)
        is_comp = any(k in s for k in comps_kw)
        if is_part and not is_comp:
            preds.append("computer_parts"); probs.append(0.90)
        elif is_comp and not is_part:
            preds.append("computers"); probs.append(0.90)
        elif is_comp and is_part:
            preds.append("computers"); probs.append(0.60)
        else:
            preds.append("computer_parts"); probs.append(0.55)
    return preds, probs

# ---------------- Load or build pipeline ----------------
@st.cache_resource(show_spinner=False)
def _load_pipeline() -> Optional[Pipeline]:
    if not MODEL_PATH.exists():
        return None
    clf = joblib.load(MODEL_PATH)

    # Sanity-check: the model should expect 25 features (our keyword featurizer provides 25)
    n_in = getattr(clf, "n_features_in_", None)
    if n_in is not None and n_in != N_FEATS:
        st.warning(
            f"Loaded model expects {n_in} features, but this app provides {N_FEATS}. "
            "If you trained on different features, rebuild the app to match."
        )
    # Wrap your LR in a pipeline with the KeywordFeaturizer so preprocessing matches training
    return Pipeline([("kw", KeywordFeaturizer(KEYWORDS)), ("clf", clf)])

pipe = _load_pipeline()

# ---------------- Tabs ----------------
tab_try, tab_bulk = st.tabs(["Try it now", "Bulk classify (CSV)"])

with tab_try:
    st.subheader("Try it now")
    default_examples = (
        "HP 24-inch monitor with HDMI, great condition\n"
        "Dell XPS 13 i7, 16GB RAM, 512GB SSD\n"
        "RTX 3060 graphics card, 12GB GDDR6, new in box\n"
        "Gaming PC Ryzen 5 5600X, 16GB DDR4, 1TB NVMe\n"
    )
    user_text = st.text_area(
        "Enter one listing per line:",
        height=160,
        placeholder="e.g., RTX 3060 graphics card, 12GB GDDR6, new in box",
        value=default_examples,
    )
    flag_thresh = st.slider("Flag if confidence below", 0.50, 0.95, 0.65, 0.01)

    if st.button("Classify", type="primary"):
        lines = [ln.strip() for ln in user_text.splitlines() if ln.strip()]
        if not lines:
            st.warning("Please enter at least one line of text.")
        else:
            if pipe is not None and hasattr(pipe.named_steps["clf"], "predict_proba"):
                preds = pipe.predict(lines)
                proba = pipe.predict_proba(lines)
                # Determine prob of 'computers' robustly for confidence
                classes = list(pipe.named_steps["clf"].classes_)
                if "computers" in classes:
                    idx_comp = classes.index("computers")
                else:
                    # pick column with higher prob on 'computer-like' probes
                    probe_comp = ["apple macbook pro 16 inch", "dell xps 13 laptop"]
                    p_comp = pipe.predict_proba(probe_comp).mean(axis=0)
                    idx_comp = int(np.argmax(p_comp))
                p_comp = proba[:, idx_comp]
                conf = np.where(preds == "computers", p_comp, 1.0 - p_comp)
            elif pipe is not None:
                # No predict_proba available; use a neutral confidence
                preds = pipe.predict(lines)
                conf = np.full(len(lines), 0.75, dtype=float)
            else:
                preds, conf = _fallback_stub(lines)

            df_out = pd.DataFrame({"text": lines, "predicted_label": preds, "confidence": np.round(conf, 3)})
            df_out["flag_low_conf"] = df_out["confidence"] < flag_thresh

            st.subheader("Results")
            st.dataframe(df_out, use_container_width=True)
            st.download_button("Download predictions", df_out.to_csv(index=False), "predictions.csv", mime="text/csv")

with tab_bulk:
    st.subheader("Bulk classify (CSV)")
    uploaded = st.file_uploader(
        "Upload a CSV",
        type=["csv"],
        help="Use a single 'text' column or both 'title' and 'description'.",
    )

    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        df_in = _normalize_input(df_raw)

        if pipe is not None and hasattr(pipe.named_steps["clf"], "predict_proba"):
            preds = pipe.predict(df_in["text"])
            proba = pipe.predict_proba(df_in["text"])
            classes = list(pipe.named_steps["clf"].classes_)
            if "computers" in classes:
                idx_comp = classes.index("computers")
            else:
                probe_comp = ["apple macbook pro 16 inch", "dell xps 13 laptop"]
                p_comp = pipe.predict_proba(probe_comp).mean(axis=0)
                idx_comp = int(np.argmax(p_comp))
            p_comp = proba[:, idx_comp]
            conf = np.where(preds == "computers", p_comp, 1.0 - p_comp)
        elif pipe is not None:
            preds = pipe.predict(df_in["text"])
            conf = np.full(len(df_in), 0.75, dtype=float)
        else:
            preds, conf = _fallback_stub(df_in["text"].tolist())

        df_out = df_in.copy()
        df_out["predicted_label"] = preds
        df_out["confidence"] = pd.Series(conf).round(3)
        st.dataframe(df_out, use_container_width=True)
        st.download_button("Download predictions", df_out.to_csv(index=False), "predictions.csv", mime="text/csv")
    else:
        st.info("No file uploaded yet.")

# Discreet footer for a non-technical audience
if pipe is not None:
    st.caption("Pre-trained model loaded. (Keyword features → Logistic Regression)")
else:
    st.caption("Demo fallback is shown because no model file was found.")
    
