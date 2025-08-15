import re
from pathlib import Path
from typing import List

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
# MODEL PATH (your saved LR trained on 25 keywords)
# ------------------------
MODEL_PATH = Path("models/final_logistic_model.pkl")

# ------------------------
# RECREATE THE 25 TRAINING KEYWORDS (ORDER MATTERS)
# ------------------------
KEYWORDS = [
    "laptop","dell","gaming","intel","core","window","pc","gb","ssd","latitude",
    "keyboard","ram","computer","mouse","mac","graphic","ryzen","monitor","cable",
    "printer","hp","router","cartridge","ink","drive"
]
N_FEATS = len(KEYWORDS)

class KeywordFeaturizer(BaseEstimator, TransformerMixin):
    """Build the exact 25 binary features your LR was trained on."""
    def __init__(self, keywords: List[str]):
        self.keywords = list(keywords)
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = ["" if x is None else str(x) for x in X]
        feats = []
        for t in X:
            s = t.lower()
            feats.append([1 if k in s else 0 for k in self.keywords])
        return np.asarray(feats, dtype=np.float32)

# ------------------------
# LOAD MODEL + ASSEMBLE PIPELINE
# ------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline() -> Pipeline:
    if not MODEL_PATH.exists():
        st.error("Model file not found. Please add `models/final_logistic_model.pkl`.")
        st.stop()
    clf = joblib.load(MODEL_PATH)

    n_in = getattr(clf, "n_features_in_", None)
    if n_in is not None and n_in != N_FEATS:
        st.error(
            f"Loaded model expects {n_in} features, but this app provides {N_FEATS}. "
            "Use the LR model trained on the 25 keyword features."
        )
        st.stop()

    pipe = Pipeline([("kw", KeywordFeaturizer(KEYWORDS)), ("clf", clf)])
    _ = pipe.predict(["sanity check"])
    return pipe

pipe = load_pipeline()

# Determine which proba column corresponds to "computers" and build class-name mapping
def resolve_label_mapping(pipeline: Pipeline):
    clf = pipeline.named_steps["clf"]
    classes = list(getattr(clf, "classes_", []))

    # If labels are strings and include 'computers', use directly
    if classes and all(isinstance(c, str) for c in classes) and "computers" in classes:
        idx_pos = classes.index("computers")
        class_to_name = {c: c for c in classes}
        return idx_pos, class_to_name

    # Otherwise infer via a quick probe
    probe_comp = ["apple macbook pro 16 inch", "dell xps 13 laptop"]
    probe_part = ["rtx 3060 graphics card", "corsair 750w psu power supply"]
    p_comp = pipeline.predict_proba(probe_comp).mean(axis=0)
    p_part = pipeline.predict_proba(probe_part).mean(axis=0)
    idx_pos = int(np.argmax(p_comp - p_part))

    if len(classes) == 2:
        other_idx = 1 - idx_pos
        class_to_name = {classes[idx_pos]: "computers", classes[other_idx]: "computer_parts"}
    else:
        class_to_name = {0: "computer_parts", 1: "computers"}  # fallback
    return idx_pos, class_to_name

IDX_POS, CLASS_TO_NAME = resolve_label_mapping(pipe)

# ------------------------
# APP TITLE & INTRO
# ------------------------
st.title("Smart Ad Classifier – Craigslist Computer Listings")
st.markdown(
    """
    Classifies Craigslist listings into **Computer** or **Computer Part** using a
    supervised NLP model trained on real marketplace data.
    """
)

# ------------------------
# PLURALITY HANDLING
# ------------------------
PLURAL_HINTS = [
    "multiple", "lot of", "lots of", "variety of", "assorted", "bundle of",
    "pair of", "pairs of", "two", "three", "four", "several", "many",
    "pcs", "pieces", "units", "items", "bulk"
]
PLURAL_NOUNS = r"\b(computers|laptops|desktops|monitors|keyboards|mice|routers|printers|drives|cables|adapters)\b"

def detect_plurality(text: str) -> bool:
    t = (text or "").lower()
    if any(h in t for h in PLURAL_HINTS):
        return True
    if re.search(PLURAL_NOUNS, t):
        return True
    return False

def pluralize_label(base_label: str, is_plural: bool) -> str:
    if base_label == "computers":
        return "Computers" if is_plural else "Computer"
    if base_label == "computer_parts":
        return "Computer Parts" if is_plural else "Computer Part"
    return base_label

# ------------------------
# HELPERS
# ------------------------
def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    """Accept 'text' OR ('title' + 'description') and return a clean DataFrame with 'text'."""
    cols = {c.lower(): c for c in df.columns}
    if "text" in cols:
        text = df[cols["text"]].fillna("").astype(str)
    elif {"title", "description"}.issubset(cols.keys()):
        text = (df[cols["title"]].fillna("").astype(str) + " " +
                df[cols["description"]].fillna("").astype(str))
    else:
        st.error("CSV must contain a 'text' column OR both 'title' and 'description'.")
        st.stop()
    out = pd.DataFrame({"text": text})
    out = out[out["text"].str.strip().ne("")].reset_index(drop=True)
    return out

def map_labels(raw_preds):
    return [CLASS_TO_NAME.get(p, str(p)) for p in raw_preds]

def predict_with_conf(texts: List[str]) -> pd.DataFrame:
    raw = pipe.predict(texts)
    base = map_labels(raw)

    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        proba = pipe.predict_proba(texts)
        p_comp = proba[:, IDX_POS]
        conf = np.where(np.array(base) == "computers", p_comp, 1.0 - p_comp)
    else:
        conf = np.full(len(texts), 0.75, dtype=float)

    # Singular vs plural display
    adjusted = [pluralize_label(lbl, detect_plurality(txt)) for lbl, txt in zip(base, texts)]

    return pd.DataFrame({
        "text": texts,
        "predicted_label": adjusted,
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
    st.markdown("Enter a product title or description to see the predicted category and confidence.")

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
            st.warning("Please enter text before classifying.")

# ------------------------
# TAB 2 – BULK CSV UPLOAD
# ------------------------
with tab2:
    st.subheader("Bulk Classification – Upload a CSV File")
    st.markdown("Upload a CSV with a **`text`** column (or **`title` + `description`**).")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            df_in = normalize_input(df_raw)

            df_out = predict_with_conf(df_in["text"].tolist())

            show = pd.concat(
                [df_in.reset_index(drop=True), df_out[["predicted_label", "confidence"]]],
                axis=1
            )
            st.dataframe(show, use_container_width=True)

            st.download_button(
                "Download Predictions",
                data=df_out.to_csv(index=False),
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
st.caption(
    "Supervised NLP model (keyword features + Logistic Regression), trained on labeled Craigslist data. "
    "Accurate, explainable, and deployment-ready."
)
