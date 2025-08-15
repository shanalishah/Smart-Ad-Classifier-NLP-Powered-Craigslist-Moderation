# streamlit_app.py — Smart Ad Classifier (showcase-ready; uses your saved 25-keyword LR model)
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# ---------------------------
# Page setup & light style
# ---------------------------
st.set_page_config(page_title="Smart Ad Classifier — Computers vs Computer Parts", layout="wide")
st.markdown(
    """
    <style>
      .smallcaps {font-variant: small-caps; letter-spacing:.03em;}
      .subtle {color:#6b7280;}
      .badge {display:inline-block; padding:4px 10px; border-radius:999px; font-size:0.85rem;}
      .badge-ok {background:#ecfdf5;}
      .badge-warn {background:#fff7ed;}
      .result {padding:10px 12px; border:1px solid #eee; border-radius:12px; margin-bottom:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Smart Ad Classifier")
st.markdown('<div class="subtle smallcaps">Automated moderation for marketplace listings</div>', unsafe_allow_html=True)

# ---------------------------
# Your 25 training keywords (ORDER matters; keep as in training)
# ---------------------------
KEYWORDS = [
    "laptop","dell","gaming","intel","core","window","pc","gb","ssd","latitude",
    "keyboard","ram","computer","mouse","mac","graphic","ryzen","monitor","cable",
    "printer","hp","router","cartridge","ink","drive"
]
N_FEATS = len(KEYWORDS)

class KeywordFeaturizer(BaseEstimator, TransformerMixin):
    """Recreate the exact 25 binary features your LR was trained on."""
    def __init__(self, keywords):
        self.keywords = list(keywords)
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        feats = []
        for t in X:
            s = str(t).lower()
            feats.append([1 if k in s else 0 for k in self.keywords])
        return np.asarray(feats, dtype=np.float32)

# ---------------------------
# Load your saved model
# ---------------------------
MODEL_PATH = Path("models/final_logistic_model.pkl")

@st.cache_resource
def load_pipeline_and_meta():
    if not MODEL_PATH.exists():
        st.error("`models/final_logistic_model.pkl` not found.")
        st.stop()
    clf = joblib.load(MODEL_PATH)
    n_in = getattr(clf, "n_features_in_", None)
    if n_in is not None and n_in != N_FEATS:
        st.error(f"Loaded model expects {n_in} features, but this app provides {N_FEATS}. "
                 "Use the LR model trained on 25 keywords.")
        st.stop()
    pipe = Pipeline([("kw", KeywordFeaturizer(KEYWORDS)), ("clf", clf)])

    # Determine which probability column corresponds to 'computers' (robustly)
    if hasattr(clf, "classes_"):
        classes = list(clf.classes_)
    else:
        classes = [0, 1]  # fallback

    # If classes are already correct strings:
    if all(isinstance(c, str) for c in classes) and "computers" in classes:
        idx_pos = classes.index("computers")
    else:
        # Infer via a quick probe: computers text should get higher prob in the positive column
        probe_comp = ["apple macbook pro 16 inch", "dell xps 13 laptop"]
        probe_part = ["rtx 3060 graphics card", "corsair 750w psu power supply"]
        p_comp = pipe.predict_proba(probe_comp).mean(axis=0)
        p_part = pipe.predict_proba(probe_part).mean(axis=0)
        # pick the column that prefers 'comp' over 'part'
        diffs = p_comp - p_part
        idx_pos = int(np.argmax(diffs))

    return pipe, idx_pos

pipe, IDX_POS = load_pipeline_and_meta()
POS_NAME, NEG_NAME = "computers", "computer_parts"

# ---------------------------
# Sidebar — brief context for recruiters
# ---------------------------
with st.sidebar:
    st.header("Project summary")
    st.markdown(
        """
        **Objective**  
        Automatically separate full **computers** from **computer parts/accessories** in marketplace listings.

        **Method**  
        Lightweight NLP classifier: 25 keyword signals → Logistic Regression.

        **Impact**  
        - Cleaner search results  
        - Less manual moderation  
        - Adaptable to other categories
        """
    )
    st.markdown("---")
    st.caption("This demo loads a pre-trained model and does not store any user data.")

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

def classify_with_conf(texts, threshold: float):
    probs = pipe.predict_proba(texts)[:, IDX_POS]
    preds = np.where(probs >= threshold, POS_NAME, NEG_NAME)
    conf = np.where(preds == POS_NAME, probs, 1.0 - probs)
    return preds, conf

def explain_keywords(texts):
    feats = pipe.named_steps["kw"].transform(texts)  # (n, 25)
    coef = pipe.named_steps["clf"].coef_[0]          # binary LR => (1, n_features)
    rows = []
    for i in range(len(texts)):
        present_idx = np.where(feats[i] == 1)[0]
        contribs = [(KEYWORDS[j], float(coef[j])) for j in present_idx]
        rows.append(sorted(contribs, key=lambda x: abs(x[1]), reverse=True)[:8])
    return rows

# ---------------------------
# Tabs: Demo | Batch | Explainability
# ---------------------------
tab_demo, tab_batch, tab_explain = st.tabs(["Demo", "Batch", "Explainability"])

# --- Demo tab
with tab_demo:
    st.subheader("Interactive Demo")
    st.caption("Enter one listing per line. The model returns a label and a confidence score.")
    samples = (
        "Apple MacBook Pro 14-inch, M2, 16GB RAM, 512GB SSD\n"
        "RTX 3060 graphics card, 12GB GDDR6, brand new\n"
        "Lenovo ThinkPad T14 laptop, 32GB RAM, 1TB SSD\n"
        "Corsair 750W PSU power supply\n"
    )
    user_text = st.text_area("Listings", value=samples, height=180, key="ta_demo")
    threshold = st.slider("Decision threshold (p=computers)", 0.10, 0.90, 0.50, 0.01, key="thr_demo")

    if st.button("Classify", type="primary", key="btn_demo"):
        lines = [ln.strip() for ln in user_text.splitlines() if ln.strip()]
        if not lines:
            st.warning("Please enter at least one line.")
        else:
            preds, conf = classify_with_conf(lines, threshold)
            df_out = pd.DataFrame({"text": lines, "predicted_label": preds, "confidence": conf.round(3)})

            for _, r in df_out.iterrows():
                badge = "badge-ok" if r["predicted_label"] == POS_NAME else "badge-warn"
                st.markdown(
                    f'<div class="result"><span class="badge {badge}">{r["predicted_label"]}</span> '
                    f'<strong>{r["confidence"]:.2f}</strong> — {r["text"]}</div>',
                    unsafe_allow_html=True
                )
            st.download_button("Download results (CSV)", df_out.to_csv(index=False), "predictions.csv", key="dl_demo")

# --- Batch tab
with tab_batch:
    st.subheader("Batch Classification (CSV)")
    st.caption("Upload a CSV with a **text** column (or **title** + **description**).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="uploader")
    sample = pd.DataFrame({
        "text": [
            "Apple iMac 27-inch 2019 i5 16GB 512GB",
            "NVIDIA RTX 3060 Ti 8GB graphics card",
            "HP Elitebook laptop 16GB RAM 512GB SSD",
            "Corsair 650W PSU power supply",
        ]
    })
    st.download_button("Download sample CSV", sample.to_csv(index=False), "sample.csv", key="dl_sample")

    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        df_in = normalize_input(df_raw)
        preds, conf = classify_with_conf(df_in["text"].tolist(), threshold=0.50)
        df_out = df_in.copy()
        df_out["predicted_label"] = preds
        df_out["confidence"] = np.round(conf, 3)
        st.dataframe(df_out, use_container_width=True)
        st.download_button("Download results (CSV)", df_out.to_csv(index=False), "predictions.csv", key="dl_batch")
    else:
        st.info("No file uploaded yet.")

# --- Explainability tab
with tab_explain:
    st.subheader("Keyword Signals Detected")
    st.caption("We show the training keywords found in each listing and the direction of their contribution.")
    txt = st.text_area("Enter one listing per line:", height=160, key="ta_explain")
    if st.button("Analyze keywords", key="btn_explain"):
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if not lines:
            st.warning("Please enter at least one line.")
        else:
            preds, conf = classify_with_conf(lines, threshold=0.50)
            contrib_rows = explain_keywords(lines)
            for i, line in enumerate(lines):
                st.markdown(f"**Text:** {line}")
                badge = "badge-ok" if preds[i] == POS_NAME else "badge-warn"
                st.markdown(
                    f'Prediction: <span class="badge {badge}">{preds[i]}</span> '
                    f'&nbsp;·&nbsp; Confidence: **{conf[i]:.2f}**',
                    unsafe_allow_html=True
                )
                contribs = contrib_rows[i]
                if not contribs:
                    st.caption("No training keywords detected in this text.")
                else:
                    df_kw = pd.DataFrame(contribs, columns=["keyword", "weight"])
                    df_kw["direction"] = np.where(df_kw["weight"] >= 0, "→ computers", "→ computer_parts")
                    st.dataframe(df_kw[["keyword", "direction", "weight"]], use_container_width=True)
                st.markdown("---")

# ---------------------------
# Footer
# ---------------------------
st.caption("Model: 25 keyword features → Logistic Regression. This demo is a showcase; results are indicative.")
