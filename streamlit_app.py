# streamlit_app.py — Smart Ad Classifier (uses unified pipeline + metadata threshold)
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Smart Ad Classifier – Craigslist Listings", layout="wide")
st.title("Smart Ad Classifier — Computers vs Computer Parts")
st.write(
    "Paste listings (one per line) to classify whether they describe a **Computer** or a **Computer Part**. "
    "You can also upload a CSV on the second tab."
)

# ---------------------------
# Model & metadata loading
# ---------------------------
ROOT = Path(".")
PIPELINE_PATHS = [
    ROOT / "pipeline_lr_tfidf.joblib",
    ROOT / "models" / "pipeline_lr_tfidf.joblib",
]
META_PATHS = [
    ROOT / "model_metadata.json",
    ROOT / "models" / "model_metadata.json",
]

@st.cache_resource
def load_pipeline():
    for p in PIPELINE_PATHS:
        if p.exists():
            pipe = joblib.load(p)
            # quick smoke test
            _ = pipe.predict(["sanity check"])
            return pipe
    return None

@st.cache_data
def load_metadata():
    meta = {"positive_class": "computers", "decision_threshold": 0.50}  # sensible defaults
    for mp in META_PATHS:
        if mp.exists():
            try:
                meta.update(json.loads(mp.read_text()))
                break
            except Exception:
                pass
    return meta

pipe = load_pipeline()
meta = load_metadata()

if pipe is None:
    st.error(
        "Model not found. Please run your local trainer to generate "
        "`pipeline_lr_tfidf.joblib` (and `model_metadata.json`) and place them in the repo root."
    )
    st.stop()

# Identify class index for the positive class ("computers" by default)
clf = pipe.named_steps.get("clf", getattr(pipe, "named_steps", {}).get("classifier", None))
classes = getattr(clf, "classes_", None)
if classes is None or len(classes) != 2:
    st.error("Unexpected classifier classes. Expected a binary classifier with two classes.")
    st.stop()

if meta["positive_class"] not in classes:
    meta["positive_class"] = classes[0]
idx_pos = list(classes).index(meta["positive_class"])
NEG_CLASS = [c for c in classes if c != meta["positive_class"]][0]
THRESH = float(meta.get("decision_threshold", 0.50))

# ---------------------------
# Helpers
# ---------------------------
def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
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

def classify(texts: list[str], threshold: float) -> pd.DataFrame:
    probs = pipe.predict_proba(texts)[:, idx_pos]
    preds = np.where(probs >= threshold, meta["positive_class"], NEG_CLASS)
    conf = np.where(preds == meta["positive_class"], probs, 1.0 - probs)
    return pd.DataFrame({"text": texts, "predicted_label": preds, "confidence": conf.round(3)})

# ---------------------------
# Sidebar (threshold override for demos)
# ---------------------------
with st.sidebar:
    st.header("Settings")
    st.caption("Loaded metadata:")
    st.write({"positive_class": meta["positive_class"], "decision_threshold": round(THRESH, 2)})
    use_custom = st.checkbox("Override decision threshold", value=False)
    custom_thr = st.slider("Custom threshold (p=computers)", 0.10, 0.90, THRESH, 0.01, disabled=not use_custom)
    active_threshold = float(custom_thr if use_custom else THRESH)

# ---------------------------
# Tabs
# ---------------------------
tab_try, tab_bulk = st.tabs(["Try It Now", "Bulk Classification (CSV)"])

# ---- Try It Now
with tab_try:
    st.subheader("Try It Now")
    default_examples = (
        "Apple MacBook Pro 14-inch, M2, 16GB RAM, 512GB SSD\n"
        "RTX 3060 graphics card, 12GB GDDR6, brand new\n"
        "Lenovo ThinkPad T14 laptop, 32GB RAM, 1TB SSD\n"
        "Corsair 750W PSU power supply\n"
        "USB-C hub with HDMI and Ethernet ports\n"
    )
    user_text = st.text_area(
        "Enter one listing per line:",
        height=200,
        value=default_examples,
        placeholder="e.g., Apple MacBook Air M2, 8GB RAM, 256GB SSD",
        key="ta_try",
    )

    if st.button("Classify", key="btn_try"):
        lines = [ln.strip() for ln in user_text.splitlines() if ln.strip()]
        if not lines:
            st.warning("Please enter at least one line of text.")
        else:
            df_out = classify(lines, active_threshold)
            df_out["flag_low_conf"] = df_out["confidence"] < 0.65  # UI hint; adjust if you like
            st.subheader("Results")
            st.dataframe(df_out, use_container_width=True)
            st.download_button(
                "Download Predictions",
                df_out.to_csv(index=False),
                "predictions.csv",
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
        df_out = classify(df_in["text"].tolist(), active_threshold)
        st.dataframe(df_out, use_container_width=True)
        st.download_button(
            "Download Predictions",
            df_out.to_csv(index=False),
            "predictions.csv",
            key="dl_predictions_bulk",
        )
    else:
        st.info("No file uploaded yet.")

# ---- (Optional) model insights
with st.expander("Model insights: most-informative features", expanded=False):
    try:
        import numpy as np
        # Try to get feature names for word+char union
        vec_union = pipe.named_steps.get("vec", None)
        feature_names = None
        if vec_union is not None and hasattr(vec_union, "transformer_list"):
            try:
                word_vec = dict(vec_union.transformer_list).get("word", vec_union.transformer_list[0][1])
                char_vec = dict(vec_union.transformer_list).get("char", vec_union.transformer_list[1][1])
                word_names = [f"w:{t}" for t in word_vec.get_feature_names_out()]
                char_names = [f"c:{t}" for t in char_vec.get_feature_names_out()]
                feature_names = np.array(word_names + char_names)
            except Exception:
                pass

        coef = clf.coef_[list(classes).index(meta["positive_class"])]
        if feature_names is not None and len(feature_names) == coef.shape[0]:
            top_pos_idx = np.argsort(coef)[-20:][::-1]
            top_neg_idx = np.argsort(coef)[:20]
            st.markdown("**Top features for predicting ‘computers’**")
            st.dataframe(
                pd.DataFrame({"feature": feature_names[top_pos_idx], "weight": np.round(coef[top_pos_idx], 4)}),
                use_container_width=True,
            )
            st.markdown("**Top features toward ‘computer_parts’**")
            st.dataframe(
                pd.DataFrame({"feature": feature_names[top_neg_idx], "weight": np.round(coef[top_neg_idx], 4)}),
                use_container_width=True,
            )
        else:
            st.info("Feature names unavailable for display (pipeline may differ).")
    except Exception as e:
        st.info(f"Feature inspection unavailable: {e}")

# ---- Footer
st.caption("Pre-trained TF–IDF + Logistic Regression pipeline loaded. Threshold taken from model_metadata.json if available.")
