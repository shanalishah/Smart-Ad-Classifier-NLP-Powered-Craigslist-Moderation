import streamlit as st, pandas as pd, joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Smart Ad Classifier", layout="wide")
st.title("🧠 Smart Ad Classifier – Computers vs Computer Parts")

MODEL = Path("pipeline_lr_tfidf.joblib")
DATA = Path("data")
HUMAN = DATA / "labeled_and_flagged_with_human_check.csv"
CLEAN = DATA / "clean_dedup_labeled.csv"

@st.cache_data(show_spinner=False)
def load_labels() -> pd.DataFrame | None:
    path = CLEAN if CLEAN.exists() else (HUMAN if HUMAN.exists() else None)
    if path is None:
        return None
    df = pd.read_csv(path)
    if "human_label" in df.columns:
        df = df.rename(columns={"human_label": "label"})
    return df[["text","label"]]

@st.cache_resource
def load_or_train_pipeline():
    # 1) Load if present
    if MODEL.exists():
        try:
            return joblib.load(MODEL)
        except Exception:
            pass
    # 2) Train from labeled CSV if available
    df = load_labels()
    if df is None or df.empty:
        return None
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=3000)),
        ("clf", LogisticRegression(max_iter=200, solver="liblinear"))
    ])
    pipe.fit(df["text"], df["label"])
    try:
        joblib.dump(pipe, MODEL)
    except Exception:
        pass
    return pipe

pipe = load_or_train_pipeline()

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Either a 'text' column OR 'title' + 'description'")
    demo = st.checkbox("Use demo samples", value=not uploaded)
    flag_thresh = st.slider("Flag if confidence below", 0.50, 0.95, 0.65, 0.01)

def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if "text" in cols:
        text = df[cols["text"]].fillna("").astype(str)
    elif {"title","description"}.issubset(set(cols)):
        text = (df[cols["title"]].fillna("") + " " + df[cols["description"]].fillna("")).astype(str)
    else:
        st.error("CSV must contain a 'text' column OR both 'title' and 'description'.")
        st.stop()
    out = pd.DataFrame({"text": text})
    out = out[out["text"].str.strip().ne("")].reset_index(drop=True)
    return out

if pipe is None:
    st.info("No trained model found. Add a labeled CSV to `data/` (e.g., `labeled_and_flagged_with_human_check.csv`) "
            "and redeploy, or upload a pre-trained `pipeline_lr_tfidf.joblib` to the repo root.")
    st.stop()

if demo:
    df_in = pd.DataFrame({
        "text":[
            "Dell Inspiron i7 16GB RAM 512GB SSD laptop – great condition",
            "HP 24-inch monitor, HDMI cable included, like new",
            "Gaming PC Ryzen 5 5600X with RTX 3060, 16GB DDR4, 1TB NVMe",
            "USB-C hub with HDMI and Ethernet ports"
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

preds = pipe.predict(df_in["text"])
proba = pipe.predict_proba(df_in["text"]).max(axis=1)

df_out = df_in.copy()
df_out["predicted_label"] = preds
df_out["confidence"] = proba.round(3)
df_out["flag_low_conf"] = df_out["confidence"] < flag_thresh

st.subheader("Predictions")
st.dataframe(df_out, use_container_width=True)
st.download_button("Download predictions", df_out.to_csv(index=False), "predictions.csv", mime="text/csv")

st.caption("TF-IDF (uni+bi, 3k) + Logistic Regression. Auto-trains if `data/` contains a labeled CSV.")
