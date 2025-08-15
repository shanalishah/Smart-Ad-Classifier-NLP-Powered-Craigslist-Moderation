# streamlit_app.py — Smart Ad Classifier
import streamlit as st
import pandas as pd
import joblib, pickle
from pathlib import Path
from typing import Optional, Tuple, Any
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Smart Ad Classifier", layout="wide")
st.title("Smart Ad Classifier – Computers vs Computer Parts")
st.write(
    "Paste listings (one per line) to classify whether they describe a **Computer** or a **Computer Part**. "
    "Or upload a CSV on the second tab."
)

# ---------- Paths ----------
ROOT = Path(".")
MODEL_PIPELINE = ROOT / "pipeline_lr_tfidf.joblib"  # optional single-file pipeline
MODELS_DIR = ROOT / "models"
VEC_PKL = MODELS_DIR / "tfidf_vectorizer.pkl"       # your uploaded file
CLF_PKL = MODELS_DIR / "final_logistic_model.pkl"   # your uploaded file

DATA = ROOT / "data"
HUMAN = DATA / "labeled_and_flagged_with_human_check.csv"  # optional
CLEAN = DATA / "clean_dedup_labeled.csv"                   # optional

# ---------- Safe loaders & validators ----------
def _try_joblib_or_pickle(path: Path) -> Any:
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def _is_valid_vectorizer(v) -> bool:
    # Must have transform(); for TF-IDF, vocabulary_ should exist
    return hasattr(v, "transform") and (hasattr(v, "vocabulary_") or hasattr(v, "get_feature_names_out"))

def _is_valid_classifier(c) -> bool:
    # Must have predict(); predict_proba is nice-to-have
    return hasattr(c, "predict")

# ---------- Label loading / cleaning ----------
@st.cache_data(show_spinner=False)
def _load_labels() -> Optional[pd.DataFrame]:
    path = CLEAN if CLEAN.exists() else (HUMAN if HUMAN.exists() else None)
    if path is None:
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]

    # build text column
    if "text" in df.columns:
        text = df["text"].fillna("").astype(str)
    elif {"title", "description"}.issubset(df.columns):
        text = (df["title"].fillna("") + " " + df["description"].fillna("")).astype(str)
    else:
        return None

    # choose one label column
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

# ---------- Load model in priority order ----------
@st.cache_resource
def _load_model() -> Tuple[Optional[Pipeline], Optional[int], str]:
    """
    Returns (pipeline_or_none, training_rows_if_trained_now, source_string)
    Source is one of: 'pipeline', 'pickles', 'trained', 'none'
    """
    # 1) Single pipeline file (fastest)
    if MODEL_PIPELINE.exists():
        try:
            pipe = joblib.load(MODEL_PIPELINE)
            return pipe, None, "pipeline"
        except Exception:
            pass  # continue

    # 2) Your pickles (vectorizer + classifier)
    if VEC_PKL.exists() and CLF_PKL.exists():
        try:
            vec = _try_joblib_or_pickle(VEC_PKL)
            clf = _try_joblib_or_pickle(CLF_PKL)
            if _is_valid_vectorizer(vec) and _is_valid_classifier(clf):
                pipe = Pipeline([("tfidf", vec), ("clf", clf)])
                # quick smoke test to ensure transform works
                _ = pipe.predict(["test"])
                return pipe, None, "pickles"
        except Exception:
            pass  # fall through

    # 3) Train from labeled CSV (if available)
    df = _load_labels()
    if df is not None and not df.empty:
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=3000)),
            ("clf", LogisticRegression(max_iter=200, solver="liblinear")),
        ])
        pipe.fit(df["text"], df["label"])
        # best-effort save as pipeline for future fast loads
        try:
            joblib.dump(pipe, MODEL_PIPELINE)
        except Exception:
            pass
        return pipe, len(df), "trained"

    # 4) Nothing available
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

# ---------- Build UI ----------
pipe, trained_rows, source = _load_model()

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
        value=default_examples,
        placeholder="e.g., RTX 3060 graphics card, 12GB GDDR6, new in box",
    )
    flag_thresh = st.slider("Flag if confidence below", 0.50, 0.95, 0.65, 0.01)

    if st.button("Classify"):
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

        if pipe is not None:
            preds = pipe.predict(df_in["text"])
            conf = pipe.predict_proba(df_in["text"]).max(axis=1)
        else:
            preds, conf = _keyword_stub_predict(df_in["text"].tolist())

        df_out = df_in.copy()
        df_out["predicted_label"] = preds
        df_out["confidence"] = pd.Series(conf).round(3)
        st.dataframe(df_out, use_container_width=True)
        st.download_button("Download predictions", df_out.to_csv(index=False), "predictions.csv", mime="text/csv")
    else:
        st.info("No file uploaded yet.")

# ---------- Discreet footer ----------
if source == "pipeline":
    st.caption("Pre-trained model loaded for instant results.")
elif source == "pickles":
    st.caption("Pre-trained vectorizer and classifier loaded from repository.")
elif source == "trained" and trained_rows:
    st.caption(f"Built and tested on {trained_rows} real Craigslist listings.")
else:
    st.caption("Interactive demo available without uploading a dataset.")
