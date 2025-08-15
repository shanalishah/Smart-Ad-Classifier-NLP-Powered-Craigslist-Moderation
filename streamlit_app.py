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

# Determine which proba column corresponds to "computers"
def resolve_idx_pos(pipeline: Pipeline) -> int:
    clf = pipeline.named_steps["clf"]
    classes = list(getattr(clf, "classes_", []))
    if classes and all(isinstance(c, str) for c in classes) and "computers" in classes:
        return classes.index("computers")
    # Probe-based inference
    probe_comp = ["apple macbook pro 16 inch", "dell xps 13 laptop"]
    probe_part = ["rtx 3060 graphics card", "corsair 750w psu power supply"]
    p_comp = pipeline.predict_proba(probe_comp).mean(axis=0)
    p_part = pipeline.predict_proba(probe_part).mean(axis=0)
    return int(np.argmax(p_comp - p_part))

IDX_POS = resolve_idx_pos(pipe)

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

def pluralize(is_computer: bool, is_plural: bool) -> str:
    if is_computer:
        return "Computers" if is_plural else "Computer"
    return "Computer Parts" if is_plural else "Computer Part"

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

# --- Lexicons for gentle, human-readable nudges ---
PARTS_STRONG = {
    "gpu","graphics card","video card","graphics","rtx","gtx","radeon",
    "psu","power supply","motherboard","mainboard","cpu","processor","heatsink","cooler",
    "ram","memory","sodimm","ddr3","ddr4","ddr5",
    "ssd","hdd","hard drive","nvme","sata","m.2",
    "monitor","screen","keyboard","mouse","mice","printer","ink","toner",
    "router","switch","cable","adapter","webcam","case","chassis","fan"
}
COMPUTERS_STRONG = {
    "laptop","notebook","macbook","mac book","imac","mac mini","desktop","tower",
    "gaming pc","prebuilt","all-in-one","aio","chromebook","computer","pc"
}

def _hit_any(text: str, vocab: set[str]) -> bool:
    s = (text or "").lower()
    return any(w in s for w in vocab)

def _apply_prior_nudge(p_comp: np.ndarray, texts: list[str]) -> np.ndarray:
    """
    Adjust p(computers) slightly when we have a strong, unambiguous keyword signal.
    - If looks like PART only -> nudge down (toward parts)
    - If looks like COMPUTER only -> nudge up (toward computers)
    Nudges are small so the model still dominates.
    """
    nudged = p_comp.copy()
    for i, t in enumerate(texts):
        part_hit = _hit_any(t, PARTS_STRONG)
        comp_hit = _hit_any(t, COMPUTERS_STRONG)
        if part_hit and not comp_hit:
            nudged[i] = max(0.0, nudged[i] - 0.20)   # 20% nudge toward parts
        elif comp_hit and not part_hit:
            nudged[i] = min(1.0, nudged[i] + 0.20)   # 20% nudge toward computers
        # both or neither -> no nudge
    return nudged

def classify_texts(texts: List[str], threshold: float) -> pd.DataFrame:
    """
    Probabilities only. p(computers) is nudged by simple rules so obvious
    items like 'mouse' flip earlier without needing an extreme threshold.
    """
    p_comp_base = pipe.predict_proba(texts)[:, IDX_POS]  # base p(computers)
    p_comp = _apply_prior_nudge(p_comp_base, texts)

    is_comp = p_comp >= threshold
    labels = [pluralize(bool(c), detect_plurality(t)) for c, t in zip(is_comp, texts)]
    conf = np.where(is_comp, p_comp, 1.0 - p_comp)

    return pd.DataFrame({
        "text": texts,
        "predicted_label": labels,
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
    threshold_demo = st.slider("Decision threshold (p = Computer)", 0.10, 0.90, 0.50, 0.01, key="thr_demo")

    user_input = st.text_area(
        "Listing text:",
        placeholder="Example: Apple MacBook Pro 16-inch, 16GB RAM, 512GB SSD"
    )

    if st.button("Classify Listing", key="btn_single_classify"):
        if user_input.strip():
            df_out = classify_texts([user_input], threshold_demo)
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
    threshold_bulk = st.slider("Decision threshold (p = Computer) for bulk", 0.10, 0.90, 0.50, 0.01, key="thr_bulk")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            df_in = normalize_input(df_raw)

            df_out = classify_texts(df_in["text"].tolist(), threshold_bulk)

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
st.caption("Supervised NLP model (keyword features + Logistic Regression), trained on labeled Craigslist data.")
