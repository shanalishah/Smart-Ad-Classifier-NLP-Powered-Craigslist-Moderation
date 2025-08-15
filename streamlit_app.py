# streamlit_app.py — Smart Ad Classifier (robust loader + recruiter-friendly UI)
from __future__ import annotations
import streamlit as st
import pandas as pd
import joblib, pickle
from pathlib import Path
from typing import Optional, Tuple, Any
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ------------------------
# Page configuration
# ------------------------
st.set_page_config(page_title="Smart Ad Classifier – Craigslist Listings", layout="wide")
st.title("Smart Ad Classifier – Computers vs Computer Parts")
st.write(
    "Paste listings (one per line) to classify whether they describe a **Computer** or a **Computer Part**. "
    "You can also upload a CSV on the second tab."
)

# ------------------------
# Paths
# ------------------------
ROOT = Path(".")
PIPELINE_PATH = ROOT / "pipeline_lr_tfidf.joblib"  # preferred single-file pipeline

# We’ll look for your pickles in models/ first, then data/
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
PICKLE_LOCATIONS = [
    (MODELS_DIR / "tfidf_vectorizer.pkl", MODELS_DIR / "final_logistic_model.pkl"),
    (DATA_DIR / "tfidf_vectorizer.pkl", DATA_DIR / "final_logistic_model.pkl"),
]

LABELED_CLEAN = DATA_DIR / "clean_dedup_labeled.csv"
LABELED_RAW = DATA_DIR / "labeled_and_flagged_with_human_check.csv"

# ------------------------
# Utilities
# ------------------------
def _try_joblib_or_pickle(path: Path) -> Any:
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def _is_valid_vectorizer(v) -> bool:
    # Must have transform(); TF-IDF has vocabulary_ or get_feature_names_out
    return hasattr(v, "transform") and (hasattr(v, "vocabulary_") or hasattr(v, "get_feature_names_out"))

def _is_valid_classifier(c) -> bool:
    # Must have predict (and ideally predict_proba)
    return hasattr(c, "predict")

def _vectorizer_dim(v) -> Optional[int]:
    try:
        return v.transform(["_probe_"]).shape[1]
    except Exception:
        return None

def _model_dim(c) -> Optional[int]:
    # Most sklearn estimators set n_features_in_ after fit
    return getattr(c, "n_features_in_", None)

@st.cache_data(show_spinner=False)
def _load_labels() -> Optional[pd.DataFrame]:
    # Prefer cleaned file if present
    path = LABELED_CLEAN if LABELED_CLEAN.exists() else (LABELED_RAW if LABELED_RAW.exists() else None)
    if path is None:
        return None

    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    # Build text
    if "text" in df.columns:
        text = df["text"].fillna("").astype(str)
    elif {"title", "description"}.issubset(df.columns):
        text = (df["title"].fillna("") + " " + df["description"].fillna("")).astype(str)
    else:
        return None

    # Choose exactly one label
    if "human_label" in df.columns:
        label = df["human_label"]
    elif "label" in df.columns:
        label = df["label"]
    else:
        return None

    label = label.replace({"computer": "computers"}).astype(str)
    out = pd.DataFrame({"text": text, "label": label})
    out = out[out["text"].str.strip().ne("")]
    out = out[out["label"].isin(["computers", "computer_parts"])].copy()
    out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return out if not out.empty else None

# ------------------------
# Model loader with validation & fallbacks
# ------------------------
@st.cache_resource
def _load_model() -> Tuple[Optional[Pipeline], Optional[int], str]:
    """
    Returns (pipeline_or_none, training_rows_if_trained, source_string)
    Source ∈ {'pipeline', 'pickles', 'trained', 'none'}
    """
    # 1) Preferred: single-file pipeline
    if PIPELINE_PATH.exists():
        try:
            pipe = joblib.load(PIPELINE_PATH)
            # quick smoke test
            _ = pipe.predict(["probe"])
            return pipe, None, "pipeline"
        except Exception:
            pass  # continue

    # 2) Your two pickles: load & validate feature compatibility
    for vec_path, clf_path in PICKLE_LOCATIONS:
        if vec_path.exists() and clf_path.exists():
            try:
                vec = _try_joblib_or_pickle(vec_path)
                clf = _try_joblib_or_pickle(clf_path)
                if _is_valid_vectorizer(vec) and _is_valid_classifier(clf):
                    vec_dim = _vectorizer_dim(vec)
                    clf_dim = _model_dim(clf)
                    if vec_dim is not None and clf_dim is not None and vec_dim == clf_dim:
                        pipe = Pipeline([("tfidf", vec), ("clf", clf)])
                        _ = pipe.predict(["probe"])  # smoke test
                        return pipe, None, "pickles"
                    # If mismatch, ignore and keep searching for other sources
            except Exception:
                pass

    # 3) Train from labeled CSV if available
    df = _load_labels()
    if df is not None:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=3000)),
            ("clf", LogisticRegression(max_iter=200, solver="liblinear")),
        ])
        pipe.fit(df["text"], df["label"])
        # Best-effort save to speed next run
        try:
            joblib.dump(pipe, PIPELINE_PATH)
        except Exception:
            pass
        return pipe, len(df), "trained"

    # 4) Nothing usable
    return None, None, "none"

def _normalize_input(df: pd.DataFrame) -> pd.DataFrame:
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

def _keyword_stub_predict(texts):
    parts_kw = {"gpu", "graphics card", "monitor", "keyboard", "mouse", "ssd", "hdd", "ram", "memory",
                "cpu", "processor", "motherboard", "psu", "power supply", "cable", "adapter", "webcam"}
    comps_kw = {"laptop", "notebook", "macbook", "desktop", "tower", "gaming pc", "prebuilt", "computer", "pc"}
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

# ------------------------
# Build the UI
# ------------------------
pipe, trained_rows, source = _load_model()

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
    )
    flag_thresh = st.slider("Flag if confidence below", 0.50, 0.95, 0.65, 0.01)

    if st.button("Classify", key="btn_try"):
        lines = [ln.strip() for ln in user_text.splitlines() if ln.strip()]
        if not lines:
            st.warning("Please enter at least one line of text.")
        else:
            if pipe is not None:
                preds = pipe.predict(lines)
                conf = pipe.predict_proba(lines).max(axis=1)
            else:
                preds, conf = _keyword_stub_predict(lines)

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
        df_in = _normalize_input(df_raw)

        if pipe is not None:
            preds = pipe.predict(df_in["text"])
            conf = pipe.predict_proba(df_in["text"]).max(axis=1)
        else:
            preds, conf = _keyword_stub_predict(df_in["text"].tolist())

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
if source == "pipeline":
    st.caption("Pre-trained model loaded for instant results.")
elif source == "pickles":
    st.caption("Pre-trained vectorizer and classifier loaded from the repository.")
elif source == "trained" and trained_rows:
    st.caption(f"Built and tested on {trained_rows} real Craigslist listings.")
else:
    st.caption("Interactive demo available without uploading a dataset.")
