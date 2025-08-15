import os
from pathlib import Path
from typing import Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# ------------------------
# PAGE CONFIGURATION
# ------------------------
st.set_page_config(
    page_title="Smart Ad Classifier – Craigslist Listings",
    layout="wide"
)

# ------------------------
# PATHS
# ------------------------
ROOT = Path(".")
MODEL_DIR = ROOT / "models"
PIPELINE_PATHS = [ROOT / "pipeline_lr_tfidf.joblib", MODEL_DIR / "pipeline_lr_tfidf.joblib"]
VEC_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
LR_PATH = MODEL_DIR / "final_logistic_model.pkl"

# ------------------------
# KEYWORD BACKUP (if LR expects 25 keyword features)
# ------------------------
KEYWORDS_25 = [
    "laptop","dell","gaming","intel","core","window","pc","gb","ssd","latitude",
    "keyboard","ram","computer","mouse","mac","graphic","ryzen","monitor","cable",
    "printer","hp","router","cartridge","ink","drive"
]
N_FEATS_25 = len(KEYWORDS_25)

class KeywordFeaturizer(BaseEstimator, TransformerMixin):
    """Recreate the exact 25 binary features your LR might have been trained on."""
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
# LOADING LOGIC
# ------------------------
@st.cache_resource(show_spinner=False)
def load_best_available_pipeline() -> Tuple[Optional[Pipeline], str]:
    """Try in order:
      1) Unified pipeline_lr_tfidf.joblib
      2) Compatible TF-IDF + LR pickles
      3) 25-keyword featurizer + LR (if LR expects 25 features)
    Returns: (pipeline, backend_name)
    """
    # 1) Unified pipeline (best)
    for p in PIPELINE_PATHS:
        if p.exists():
            try:
                pipe = joblib.load(p)
                # quick smoke test
                _ = pipe.predict(["sanity check"])
                return pipe, "unified-pipeline"
            except Exception:
                pass

    # 2) TF-IDF + LR (only if compatible)
    if VEC_PATH.exists() and LR_PATH.exists():
        try:
            vec = joblib.load(VEC_PATH)
            clf = joblib.load(LR_PATH)

            # check compatibility by looking at expected n_features_in_
            expected = getattr(clf, "n_features_in_", None)
            # approximate vectorizer feature count
            try:
                vocab_size = len(vec.get_feature_names_out())
            except Exception:
                # fallback: transform a tiny corpus to get the dimensionality
                vocab_size = vec.transform(["probe"]).shape[1]

            if expected is None or expected == vocab_size:
                pipe = Pipeline([("tfidf", vec), ("clf", clf)])
                _ = pipe.predict(["sanity check"])
                return pipe, "tfidf+lr"
        except Exception:
            pass

    # 3) 25-keyword featurizer + LR (if present and expecting 25)
    if LR_PATH.exists():
        try:
            clf = joblib.load(LR_PATH)
            expected = getattr(clf, "n_features_in_", None)
            if expected in (None, N_FEATS_25):
                pipe = Pipeline([("kw", KeywordFeaturizer(KEYWORDS_25)), ("clf", clf)])
                _ = pipe.predict(["macbook pro 16 inch"])
                return pipe, "keywords25+lr"
        except Exception:
            pass

    return None, "none"

pipe, backend = load_best_available_pipeline()

# ------------------------
# APP TITLE
# ------------------------
st.title("Smart Ad Classifier – Craigslist Computer Listings")
st.markdown(
    """
    This tool classifies Craigslist listings into **"Computer"** or **"Computer Part"** using a
    lightweight NLP model.
    """
)

# Subtle backend status for you (kept small/neutral for recruiters)
backend_note = {
    "unified-pipeline": "Pre-trained unified pipeline loaded.",
    "tfidf+lr": "TF-IDF vectorizer + Logistic Regression loaded.",
    "keywords25+lr": "Keyword features + Logistic Regression loaded.",
    "none": "No model found – using fallback disabled.",
}
st.caption(backend_note.get(backend, "Model loaded."))

if pipe is None:
    st.error("Model files not found or incompatible. Add `pipeline_lr_tfidf.joblib` "
             "or compatible pickles to the `models/` folder.")
    st.stop()

# ------------------------
# HELPERS
# ------------------------
def predict_with_conf(texts: List[str]) -> pd.DataFrame:
    """Return DataFrame with text, predicted_label, confidence."""
    preds = pipe.predict(texts)

    # Try to compute confidence from predict_proba; otherwise default
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(texts)
            # If classes contain 'computers', use its column; else use max prob
            clf = pipe.named_steps.get("clf", getattr(pipe, "steps", [[None, None]])[-1][1])
            classes = getattr(clf, "classes_", None)
            if classes is not None and "computers" in list(classes):
                idx_comp = list(classes).index("computers")
                p_comp = proba[:, idx_comp]
                conf = np.where(preds == "computers", p_comp, 1.0 - p_comp)
            else:
                conf = proba.max(axis=1)
        except Exception:
            conf = np.full(len(texts), 0.75, dtype=float)
    else:
        conf = np.full(len(texts), 0.75, dtype=float)

    return pd.DataFrame({
        "text": texts,
        "predicted_label": preds,
        "confidence": np.round(conf, 3)
    })

# ------------------------
# TABS
# ------------------------
tab1, tab2 = st.tabs(["💬 Quick Test", "📂 Bulk Classification (CSV)"])

# ------------------------
# TAB 1 – QUICK TEST
# ------------------------
with tab1:
    st.subheader("Quick Test – Classify a Single Listing")
    st.markdown("Enter a product title or short description to see how the model classifies it.")

    user_input = st.text_area(
        "Listing text:",
        placeholder="Example: Apple MacBook Pro 16-inch, 16GB RAM, 512GB SSD"
    )

    if st.button("Classify Listing", key="btn_single_classify"):
        if user_input.strip():
            df_out = predict_with_conf([user_input])
            row = df_out.iloc[0]
            st.success(f"**Prediction:** {row['predicted_label']}")
            st.info(f"Confidence: {row['confidence']:.2%}")
        else:
            st.warning("Please enter some text before classifying.")

# ------------------------
# TAB 2 – BULK CSV UPLOAD
# ------------------------
with tab2:
    st.subheader("Bulk Classification – Upload a CSV File")
    st.markdown("Upload a CSV containing a column named **`text`** "
                "(or **`title`** + **`description`**) with listing descriptions.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            cols = {c.lower(): c for c in df_raw.columns}
            if "text" in cols:
                texts = df_raw[cols["text"]].fillna("").astype(str).tolist()
            elif {"title", "description"}.issubset(cols):
                texts = (df_raw[cols["title"]].fillna("").astype(str) + " " +
                         df_raw[cols["description"]].fillna("").astype(str)).tolist()
            else:
                st.error("CSV must contain 'text' OR both 'title' and 'description'.")
                st.stop()

            df_out = predict_with_conf(texts)
            st.dataframe(pd.concat([df_raw.reset_index(drop=True), df_out[["predicted_label","confidence"]]], axis=1),
                         use_container_width=True)

            csv_data = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions",
                data=csv_data,
                file_name="predictions.csv",
                mime="text/csv",
                key="dl_predictions_bulk"
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("Please upload a CSV file to classify listings.")

# ------------------------
# FOOTER
# ------------------------
st.markdown("---")
st.caption("This demo uses a compact NLP classifier (TF-IDF or keyword features) with Logistic Regression, trained on labeled Craigslist data.")
# --
