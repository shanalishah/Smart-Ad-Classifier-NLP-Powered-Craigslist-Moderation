# app.py  — Smart Ad Classifier (Streamlit)
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Smart Ad Classifier", layout="wide")
st.title("🧠 Smart Ad Classifier – Computers vs Computer Parts")
st.caption("Upload a CSV to get predictions. If a labeled CSV is in /data, the model auto-trains on first run.")

# Paths
MODEL = Path("pipeline_lr_tfidf.joblib")
DATA = Path("data")
HUMAN = DATA / "labeled_and_flagged_with_human_check.csv"  # your labeled file
CLEAN = DATA / "clean_dedup_labeled.csv"                   # optional cleaned file (preferred if present)

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_labels() -> pd.DataFrame | None:
    """
    Return a clean DataFrame with exactly: columns ['text', 'label'].
    Accepts either:
      - CLEAN (clean_dedup_labeled.csv) with columns text + human_label/label
      - HUMAN (labeled_and_flagged_with_human_check.csv) with text or (title+description) + human_label
    Ensures a single 1D 'label' Series, valid classes, and no duplicate/blank text.
    """
    path = CLEAN if CLEAN.exists() else (HUMAN if HUMAN.exists() else None)
    if path is None:
        return None

    df = pd.read_csv(path)
    # normalize headers
    df.columns = [c.lower() for c in df.columns]

    # build text
    if "text" in df.columns:
        text = df["text"].fillna("").astype(str)
    elif {"title", "description"}.issubset(df.columns):
        text = (df["title"].fillna("") + " " + df["description"].fillna("")).astype(str)
    else:
        # no usable text columns
        return None

    # choose exactly one label column
    if "human_label" in df.columns:
        label = df["human_label"]
    elif "label" in df.columns:
        label = df["label"]
    else:
        return None

    # normalize labels
    label = label.replace({"computer": "computers"}).astype(str)

    # construct clean frame
    out = pd.DataFrame({"text": text, "label": label})
    out = out[out["text"].str.strip().ne("")]
    out = out[out["label"].isin(["computers", "computer_parts"])].copy()
    out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return out if not out.empty else None


@st.cache_resource
def load_or_train_pipeline():
    """
    Load a pre-trained pipeline if present; otherwise train from labeled CSV (if available).
    Saves the trained pipeline to MODEL when possible.
    """
    # Load existing model if present
    if MODEL.exists():
        try:
            return joblib.load(MODEL)
        except Exception:
            pass  # fall back to train

    # Train if labeled data available
    df = load_labels()
    if df is None or df.empty:
        return None

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=3000)),
        ("clf", LogisticRegression(max_iter=200, solver="liblinear")),
    ])
    pipe.fit(df["text"], df["label"])

    # Best-effort save (okay if fails on read-only FS)
    try:
        joblib.dump(pipe, MODEL)
    except Exception:
        pass

    return pipe


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept either a 'text' column OR ('title' + 'description') and produce a clean DataFrame with 'text'.
    """
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


# ---------- App ----------
pipe = load_or_train_pipeline()

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Either a 'text' column OR both 'title' and 'description'.",
    )
    demo = st.checkbox("Use demo samples", value=not uploaded)
    flag_thresh = st.slider("Flag if confidence below", 0.50, 0.95, 0.65, 0.01)

# If we have neither model nor data to train, guide user
if pipe is None:
    st.info(
        "No trained model found. To enable training on deploy, upload a labeled CSV to `data/` "
        "(e.g., `labeled_and_flagged_with_human_check.csv`) with columns:\n\n"
        "- `text` **or** both `title` + `description`\n"
        "- `human_label` (values: `computers`, `computer_parts`)\n\n"
        "Alternatively, commit a pre-trained `pipeline_lr_tfidf.joblib` to the repo root."
    )
    st.stop()

# Prepare input
if demo:
    df_in = pd.DataFrame({
        "text": [
            "Dell Inspiron i7 16GB RAM 512GB SSD laptop – great condition",
            "HP 24-inch monitor, HDMI cable included, like new",
            "Gaming PC Ryzen 5 5600X with RTX 3060, 16GB DDR4, 1TB NVMe",
            "USB-C hub with HDMI and Ethernet ports",
        ]
    })
elif uploaded:
    tmp = pd.read_csv(uploaded)
    df_in = normalize_input(tmp)
else:
    st.info("Upload a CSV or enable demo samples in the sidebar.")
    st.stop()

st.subheader("Input Preview")
st.dataframe(df_in.head(20), use_container_width=True)

# Predict
preds = pipe.predict(df_in["text"])
proba = pipe.predict_proba(df_in["text"]).max(axis=1)

df_out = df_in.copy()
df_out["predicted_label"] = preds
df_out["confidence"] = proba.round(3)
df_out["flag_low_conf"] = df_out["confidence"] < flag_thresh

st.subheader("Predictions")
st.dataframe(df_out, use_container_width=True)

st.download_button(
    "Download predictions",
    df_out.to_csv(index=False),
    "predictions.csv",
    mime="text/csv",
)

st.caption("Model: TF-IDF (uni+bi, 3k) + Logistic Regression. Trains automatically if `data/` contains a labeled CSV.")
