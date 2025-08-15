# app.py — Smart Ad Classifier (Streamlit)
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Smart Ad Classifier", layout="wide")
st.title("Smart Ad Classifier – Computers vs Computer Parts")
st.write(
    "Paste a listing (or several, one per line) to classify whether it describes a "
    "**Computer** or a **Computer Part**. You can also upload a CSV on the second tab."
)

# Paths
MODEL = Path("pipeline_lr_tfidf.joblib")
DATA = Path("data")
HUMAN = DATA / "labeled_and_flagged_with_human_check.csv"  # optional
CLEAN = DATA / "clean_dedup_labeled.csv"                   # optional

# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False)
def _load_labels() -> Optional[pd.DataFrame]:
    """
    Return a clean DataFrame with exactly: ['text', 'label'].
    Accepts either:
      - CLEAN (text + human_label/label), or
      - HUMAN (text or title+description + human_label)
    """
    path = CLEAN if CLEAN.exists() else (HUMAN if HUMAN.exists() else None)
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

    # Choose exactly one label column
    if "human_label" in df.columns:
        label = df["human_label"]
    elif "label" in df.columns:
        label = df["label"]
    else:
        return None

    # Normalize & clean
    label = label.replace({"computer": "computers"}).astype(str)
    out = pd.DataFrame({"text": text, "label": label})
    out = out[out["text"].str.strip().ne("")]
    out = out[out["label"].isin(["computers", "computer_parts"])].copy()
    out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return out if not out.empty else None


@st.cache_resource
def _load_or_train_pipeline() -> Tuple[Pipeline, Optional[int]]:
    """
    Load a saved pipeline if present; otherwise train from labeled CSV if available.
    Returns (pipeline, training_rows) where training_rows is None if not trained now.
    """
    # 1) Load saved model if present
    if MODEL.exists():
        try:
            pipe = joblib.load(MODEL)
            return pipe, None
        except Exception:
            pass  # fall through to training

    # 2) Train from labeled CSV if available
    df = _load_labels()
    if df is None or df.empty:
        # No model and no data → return None later, handled by fallback stub
        return None, None  # type: ignore

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=3000)),
        ("clf", LogisticRegression(max_iter=200, solver="liblinear")),
    ])
    pipe.fit(df["text"], df["label"])

    # Best-effort save
    try:
        joblib.dump(pipe, MODEL)
    except Exception:
        pass

    return pipe, len(df)


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


def _keyword_stub_predict(texts):
    """
    Fallback when no model/data is available: a tiny keyword heuristic so the demo still works.
    Returns preds, probs (list[str], list[float]).
    """
    parts_kw = {"gpu", "graphics card", "monitor", "keyboard", "mouse", "ssd", "hdd", "ram", "memory",
                "cpu", "processor", "motherboard", "psu", "power supply", "cable", "adapter", "webcam"}
    comps_kw = {"laptop", "notebook", "macbook", "desktop", "tower", "gaming pc", "prebuilt", "computer"}
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
            preds.append("computers"); probs.append(0.60)   # default bias
        else:
            preds.append("computer_parts"); probs.append(0.55)
    return preds, probs


# ---------------- UI ----------------
pipe, trained_rows = _load_or_train_pipeline()

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

        if pipe is None:
            # Use stub if no trained model
            preds, conf = _keyword_stub_predict(df_in["text"].tolist())
        else:
            preds = pipe.predict(df_in["text"])
            conf = pipe.predict_proba(df_in["text"]).max(axis=1)

        df_out = df_in.copy()
        df_out["predicted_label"] = preds
        df_out["confidence"] = pd.Series(conf).round(3)
        st.dataframe(df_out, use_container_width=True)
        st.download_button("Download predictions", df_out.to_csv(index=False), "predictions.csv", mime="text/csv")
    else:
        st.info("No file uploaded yet.")

# Discreet footer for a non-technical audience
if trained_rows:
    st.caption(f"Model trained on {trained_rows} labeled listings.")
else:
    st.caption("Demo is available even without a pre-trained model.")
