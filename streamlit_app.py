# streamlit_app.py — Smart Ad Classifier (fixes label mapping for your saved model)
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import joblib, pickle
from pathlib import Path
from typing import List, Optional, Dict
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
    """Build the exact 25 binary keyword features used in training."""
    def __init__(self, keywords: List[str]):
        self.keywords = list(keywords)
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        feats = []
        for t in X:
            s = str(t).lower()
            feats.append([1 if k in s else 0 for k in self.keywords])
        return np.asarray(feats, dtype=np.float32)

# ------------------------
# Load your pickled LogisticRegression
# ------------------------
ROOT = Path(".")
MODEL_PATHS = [
    ROOT / "models" / "final_logistic_model.pkl",
    ROOT / "final_logistic_model.pkl",
]

def load_clf():
    for p in MODEL_PATHS:
        if p.exists():
            try:
                try:
                    return joblib.load(p)
                except Exception:
                    with open(p, "rb") as f:
                        return pickle.load(f)
            except Exception:
                continue
    return None

clf = load_clf()
if clf is None:
    st.error("Model file not found. Please add `models/final_logistic_model.pkl` to the repository.")
    st.stop()

n_in = getattr(clf, "n_features_in_", None)
if n_in is not None and n_in != N_FEATS:
    st.error(f"Loaded model expects {n_in} features, but the keyword featurizer provides {N_FEATS}. "
             "Please ensure this is the final keyword-based model.")
    st.stop()

# Runtime pipeline: [KeywordFeaturizer] -> [Your LogisticRegression]
pipe = Pipeline([("kw", KeywordFeaturizer(KEYWORDS)), ("clf", clf)])

# ------------------------
# Infer correct label mapping (critical fix)
# ------------------------
# The model likely has numeric classes (e.g., array([0, 1])) but we need to map: which is 'computers'?
classes = getattr(clf, "classes_", None)
if classes is None or len(classes) != 2:
    st.error("Unexpected model `classes_`. Expected a binary classifier.")
    st.stop()

# If they are already strings, and match expected names, use directly.
if all(isinstance(c, str) for c in classes) and set(classes) == {"computers", "computer_parts"}:
    class_to_name: Dict = {c: c for c in classes}  # identity mapping
    name_to_index: Dict[str, int] = {c: i for i, c in enumerate(classes)}
else:
    # We must infer which numeric class id corresponds to "computers" vs "computer_parts".
    probe_computers = [
        "apple macbook pro 16 inch m1 pro",
        "dell xps 13 i7 16gb ram 512gb ssd",
        "gaming pc ryzen 5 rtx 3060 16gb ram",
        "lenovo thinkpad t14 laptop",
        "imac 27 inch retina"
    ]
    probe_parts = [
        "rtx 3060 graphics card 12gb",
        "16gb ddr4 ram kit",
        "1tb nvme ssd",
        "intel i7 9700k cpu processor",
        "corsair 750w psu power supply"
    ]
    y_comp = pipe.predict(probe_computers)
    y_part = pipe.predict(probe_parts)

    # Choose the class id that most commonly appears for computer examples
    # as 'computers', the other as 'computer_parts'.
    unique = list(classes)
    score = {unique[0]: (y_comp == unique[0]).sum(), unique[1]: (y_comp == unique[1]).sum()}
    computers_id = max(score, key=score.get)
    parts_id = unique[0] if computers_id == unique[1] else unique[1]

    class_to_name = {computers_id: "computers", parts_id: "computer_parts"}
    name_to_index = {"computers": list(classes).index(computers_id),
                     "computer_parts": list(classes).index(parts_id)}

# Helper to map raw predictions to names
def map_preds_to_names(raw_preds):
    return [class_to_name.get(p, str(p)) for p in raw_preds]

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

def predicted_confidences(texts: List[str]) -> np.ndarray:
    """Confidence of the predicted class for each text (probability if available)."""
    if hasattr(clf, "predict_proba"):
        probs = pipe.predict_proba(texts)  # columns correspond to `classes`
        # For each row, pick the probability of the predicted class
        raw_preds = pipe.predict(texts)
        idx = [list(classes).index(p) for p in raw_preds]
        return probs[np.arange(len(texts)), idx]
    else:
        # If model lacks predict_proba, return a neutral constant
        return np.full(len(texts), 0.75, dtype=float)

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
            raw = pipe.predict(lines)                   # may be numeric or string classes
            labels = map_preds_to_names(raw)            # map to canonical names
            conf = predicted_confidences(lines)

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

        raw = pipe.predict(df_in["text"])
        labels = map_preds_to_names(raw)
        conf = predicted_confidences(df_in["text"].tolist())

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
st.caption("Pre-trained logistic model loaded with the original keyword features (label mapping inferred automatically).")
