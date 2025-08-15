# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Smart Ad Classifier – Craigslist Listings", layout="wide")
st.title("Smart Ad Classifier – Computers vs Computer Parts")
st.write("Paste listings (one per line) or upload a CSV to classify them.")

# --- Paths ---
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "final_logistic_model.pkl"
VEC_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

# --- Load model and vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
    if not MODEL_PATH.exists() or not VEC_PATH.exists():
        st.error("Model or vectorizer not found in 'models/' folder.")
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# --- Normalize Input ---
def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if "text" in cols:
        text = df[cols["text"]].fillna("").astype(str)
    elif {"title", "description"}.issubset(set(cols)):
        text = (df[cols["title"]].fillna("") + " " + df[cols["description"]].fillna("")).astype(str)
    else:
        st.error("CSV must contain 'text' OR both 'title' and 'description'.")
        st.stop()
    out = pd.DataFrame({"text": text})
    out = out[out["text"].str.strip().ne("")].reset_index(drop=True)
    return out

# --- Tabs ---
tab_try, tab_bulk = st.tabs(["Try It Now", "Bulk Classification (CSV)"])

with tab_try:
    st.subheader("Try It Now")
    default_examples = (
        "Apple MacBook Pro 14-inch, M2, 16GB RAM, 512GB SSD\n"
        "RTX 3060 graphics card, 12GB, brand new\n"
        "Lenovo ThinkPad T14 laptop, 32GB RAM, 1TB SSD\n"
        "Corsair 750W PSU power supply\n"
    )
    user_text = st.text_area("Enter one listing per line:", height=180, value=default_examples)
    flag_thresh = st.slider("Flag if confidence below", 0.50, 0.95, 0.65, 0.01)

    if st.button("Classify"):
        if model is None or vectorizer is None:
            st.error("Model not loaded.")
        else:
            lines = [ln.strip() for ln in user_text.splitlines() if ln.strip()]
            X_tfidf = vectorizer.transform(lines)
            preds = model.predict(X_tfidf)
            conf = model.predict_proba(X_tfidf).max(axis=1)
            df_out = pd.DataFrame({"text": lines, "predicted_label": preds, "confidence": conf.round(3)})
            df_out["flag_low_conf"] = df_out["confidence"] < flag_thresh
            st.dataframe(df_out, use_container_width=True)
            st.download_button("Download Predictions", df_out.to_csv(index=False), "predictions.csv")

with tab_bulk:
    st.subheader("Bulk Classification (CSV)")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"], help="Use a 'text' column or both 'title' and 'description'.")
    if uploaded is not None:
        if model is None or vectorizer is None:
            st.error("Model not loaded.")
        else:
            df_raw = pd.read_csv(uploaded)
            df_in = normalize_input(df_raw)
            X_tfidf = vectorizer.transform(df_in["text"])
            preds = model.predict(X_tfidf)
            conf = model.predict_proba(X_tfidf).max(axis=1)
            df_out = df_in.copy()
            df_out["predicted_label"] = preds
            df_out["confidence"] = conf.round(3)
            st.dataframe(df_out, use_container_width=True)
            st.download_button("Download Predictions", df_out.to_csv(index=False), "predictions.csv")
    else:
        st.info("No file uploaded yet.")

st.caption("TF–IDF + Logistic Regression pipeline loaded for instant results.")
