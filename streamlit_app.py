# streamlit_app.py — uses your saved 25-keyword LogisticRegression
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Smart Ad Classifier – Craigslist Listings", layout="wide")
st.title("Smart Ad Classifier — Computers vs Computer Parts")
st.write("Paste listings (one per line) or upload a CSV to classify them.")

# --------------------------------------------------
# Your original 25 keywords (order must match TRAINING)
# --------------------------------------------------
KEYWORDS = [
    "laptop","dell","gaming","intel","core","window","pc","gb","ssd","latitude",
    "keyboard","ram","computer","mouse","mac","graphic","ryzen","monitor","cable",
    "printer","hp","router","cartridge","ink","drive"
]
N_FEATS = len(KEYWORDS)

class KeywordFeaturizer(BaseEstimator, TransformerMixin):
    """Build the exact 25 binary features used during training."""
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

# --------------------------------------------------
# Load your saved LogisticRegression
# --------------------------------------------------
MODEL_PATH = Path("models/final_logistic_model.pkl")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error("Missing models/final_logistic_model.pkl")
        st.stop()
    clf = joblib.load(MODEL_PATH)
    # sanity check: it should expect 25 features
    n_in = getattr(clf, "n_features_in_", None)
    if n_in is not None and n_in != N_FEATS:
        st.error(f"Loaded model expects {n_in} features but keyword featurizer provides {N_FEATS}. "
                 "Use the original final_logistic_model.pkl that was trained on 25 keywords.")
        st.stop()
    return clf

clf = load_model()

# Build runtime pipeline: [KeywordFeaturizer] -> [Your LogisticRegression]
pipe = Pipeline([("kw", KeywordFeaturizer(KEYWORDS)), ("clf", clf)])

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if "text" in cols:
        text = df[cols["text"]].fillna("").astype(str)
    elif {"title", "description"}.issubset(set(cols)):
        text = (df[cols["title"]].fillna("") + " " + df[cols["description"]].fillna("")).astype(str)
    else:
        st.error("CSV must contain a 'text' column OR both 'title' and 'description'.")
        st.stop()
    out = pd.DataFrame({"text": text})
    out = out[out["text"].str.strip().ne("")].reset_index(drop=True)
    return out

# Map raw predictions (could be 0/1) to canonical names
classes = getattr(clf, "classes_", None)
if classes is None:
    # fallback if classes_ missing
    CLASS_NAMES = ["computer_parts", "computers"]
else:
    # If model stored numeric classes, we’ll infer names via a small probe
    if all(isinstance(c, str) for c in classes) and set(classes) == {"computers","computer_parts"}:
        CLASS_NAMES = list(classes)
    else:
        # Probe a couple of obvious texts to decide which id means which label
        probe_comp = ["apple macbook pro 16 inch", "dell xps 13 laptop"]
        probe_part = ["rtx 3060 graphics card", "corsair 750w psu power supply"]
        pred_comp = pipe.predict(probe_comp)
        pred_part = pipe.predict(probe_part)
        uniq = list(classes)
        score = {uniq[0]: (pred_comp == uniq[0]).sum(), uniq[1]: (pred_comp == uniq[1]).sum()}
        comp_id = max(score, key=score.get)
        part_id = uniq[0] if comp_id == uniq[1] else uniq[1]
        CLASS_NAMES = ["computers" if c == comp_id else "computer_parts" for c in classes]

POS_CLASS = "computers"
NEG_CLASS = "computer_parts"

def to_labels(raw_preds):
    # raw_preds may be 0/1 or class ids; map to strings using CLASS_NAMES order
    if set(CLASS_NAMES) == {"computers","computer_parts"}:
        # Build id->name map
        id_to_name = {}
        for i, c in enumerate(classes):
            id_to_name[c] = CLASS_NAMES[i]
        return [id_to_name.get(p, str(p)) for p in raw_preds]
    return [str(p) for p in raw_preds]

def confidences(texts, labels):
    if hasattr(clf, "predict_proba"):
        probs = pipe.predict_proba(texts)
        # find column for "computers"
        if set(CLASS_NAMES) == {"computers","computer_parts"}:
            idx_comp = CLASS_NAMES.index("computers")
        else:
            idx_comp = 1 if classes[1] == "computers" else 0
        p_comp = probs[:, idx_comp]
        conf = np.where(np.array(labels) == "computers", p_comp, 1.0 - p_comp)
        return conf
    # fallback: constant confidence
    return np.full(len(texts), 0.75, dtype=float)

# --------------------------------------------------
# UI
# --------------------------------------------------
tab_try, tab_bulk = st.tabs(["Try It Now", "Bulk Classification (CSV)"])

with tab_try:
    st.subheader("Try It Now")
    default_examples = (
        "Apple MacBook Pro 14-inch, M2, 16GB RAM, 512GB SSD\n"
        "RTX 3060 graphics card, 12GB GDDR6, brand new\n"
        "Lenovo ThinkPad T14 laptop, 32GB RAM, 1TB SSD\n"
        "Corsair 750W PSU power supply\n"
    )
    user_text = st.text_area("Enter one listing per line:", height=180, value=default_examples)
    flag_thresh = st.slider("Flag if confidence below", 0.50, 0.95, 0.65, 0.01)

    if st.button("Classify", key="btn_try"):
        lines = [ln.strip() for ln in user_text.splitlines() if ln.strip()]
        if not lines:
            st.warning("Please enter at least one line of text.")
        else:
            raw = pipe.predict(lines)
            labels = to_labels(raw)
            conf = confidences(lines, labels).round(3)
            df_out = pd.DataFrame({"text": lines, "predicted_label": labels, "confidence": conf})
            df_out["flag_low_conf"] = df_out["confidence"] < flag_thresh
            st.dataframe(df_out, use_container_width=True)
            st.download_button("Download Predictions", df_out.to_csv(index=False), "predictions.csv", key="dl_predictions_try")

with tab_bulk:
    st.subheader("Bulk Classification (CSV)")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"], help="Use a 'text' column or both 'title' and 'description'.")
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        df_in = normalize_input(df_raw)
        raw = pipe.predict(df_in["text"])
        labels = to_labels(raw)
        conf = confidences(df_in["text"].tolist(), labels).round(3)
        df_out = df_in.copy()
        df_out["predicted_label"] = labels
        df_out["confidence"] = conf
        st.dataframe(df_out, use_container_width=True)
        st.download_button("Download Predictions", df_out.to_csv(index=False), "predictions.csv", key="dl_predictions_bulk")
    else:
        st.info("No file uploaded yet.")

st.caption("Using your saved keyword-based Logistic Regression. Tip: remove models/tfidf_vectorizer.pkl to avoid confusion.")
