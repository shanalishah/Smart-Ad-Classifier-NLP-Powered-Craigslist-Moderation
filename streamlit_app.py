# streamlit_app.py — Smart Ad Classifier (showcase-ready, using your saved keyword model)
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# ---------------------------
# Page setup & minimal styling
# ---------------------------
st.set_page_config(page_title="Smart Ad Classifier — Computers vs Computer Parts", layout="wide")
st.markdown(
    """
    <style>
      .smallcaps {font-variant: small-caps; letter-spacing: .03em;}
      .subtle {color:#6b7280;}
      .badge {display:inline-block; padding:4px 10px; border-radius:999px; font-size:0.85rem;}
      .badge-ok {background:#ecfdf5;}
      .badge-warn {background:#fff7ed;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Smart Ad Classifier")
st.markdown('<div class="subtle smallcaps">Automated moderation for marketplace listings</div>', unsafe_allow_html=True)

# ---------------------------
# Your original 25 keywords (order MUST match training)
# ---------------------------
KEYWORDS = [
    "laptop","dell","gaming","intel","core","window","pc","gb","ssd","latitude",
    "keyboard","ram","computer","mouse","mac","graphic","ryzen","monitor","cable",
    "printer","hp","router","cartridge","ink","drive"
]
N_FEATS = len(KEYWORDS)

class KeywordFeaturizer(BaseEstimator, TransformerMixin):
    """Rebuild the exact 25 binary features used at training time."""
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
# Load your saved LogisticRegression
# ---------------------------
MODEL_PATH = Path("models/final_logistic_model.pkl")

@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        st.error("`models/final_logistic_model.pkl` not found.")
        st.stop()
    clf = joblib.load(MODEL_PATH)

    # Sanity: it should have been trained on 25 features
    n_in = getattr(clf, "n_features_in_", None)
    if n_in is not None and n_in != N_FEATS:
        st.error(f"Loaded model expects {n_in} features, but the keyword featurizer provides {N_FEATS}. "
                 "Please use the original model trained on 25 keywords.")
        st.stop()

    # Build a runtime pipeline: features → classifier
    return Pipeline([("kw", KeywordFeaturizer(KEYWORDS)), ("clf", clf)])

pipe = load_pipeline()
clf = pipe.named_steps["clf"]

# Figure out label names and positive class index
classes = getattr(clf, "classes_", None)
if classes is None:
    # Fallback: assume binary {0,1} where 1 = computers
    CLASS_MAP = {0: "computer_parts", 1: "computers"}
else:
    # If strings already present, use them; otherwise infer
    if all(isinstance(c, str) for c in classes) and set(classes) == {"computers", "computer_parts"}:
        CLASS_MAP = {c: c for c in classes}
    else:
        # Probe to infer mapping
        probe_comp = ["apple macbook pro 16 inch", "dell xps 13 laptop"]
        probe_part = ["rtx 3060 graphics card", "corsair 750w psu"]
        y_comp = pipe.predict(probe_comp)
        uniq = list(classes)
        score = {uniq[0]: (y_comp == uniq[0]).sum(), uniq[1]: (y_comp == uniq[1]).sum()}
        comp_id = max(score, key=score.get)
        part_id = uniq[0] if comp_id == uniq[1] else uniq[1]
        CLASS_MAP = {comp_id: "computers", part_id: "computer_parts"}

# Helper: map raw preds to canonical strings
def to_labels(raw_preds):
    return [CLASS_MAP.get(p, str(p)) for p in raw_preds]

# Compute confidence as probability of the chosen class
def confidences(texts, labels):
    if hasattr(clf, "predict_proba"):
        probs = pipe.predict_proba(texts)
        # Proba column for "computers"
        if all(isinstance(c, str) for c in classes or []):
            idx_comp = list(classes).index("computers") if "computers" in classes else 1
        else:
            # If classes are not strings, derive from CLASS_MAP
            # Find the index in classes that maps to "computers"
            idx_comp = 1 if CLASS_MAP.get(classes[1], None) == "computers" else 0
        p_comp = probs[:, idx_comp]
        conf = np.where(np.array(labels) == "computers", p_comp, 1.0 - p_comp)
        return conf
    return np.full(len(texts), 0.75, dtype=float)

# Explainability: show keyword contributions (present keywords × LR weights)
def explain_rows(texts):
    feats = pipe.named_steps["kw"].transform(texts)        # (n, 25)
    coef = clf.coef_[0]                                    # binary LR => (1, n_features)
    rows = []
    for i, s in enumerate(texts):
        present = np.where(feats[i] == 1)[0]
        contribs = [(KEYWORDS[j], float(coef[j])) for j in present]
        contribs_sorted = sorted(contribs, key=lambda x: abs(x[1]), reverse=True)
        rows.append(contribs_sorted[:8])  # top 8 contributions for readability
    return rows

# ---------------------------
# Sidebar — concise project context
# ---------------------------
with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
        **Goal**  
        Automatically separate full **computers** from **computer parts/accessories** in marketplace listings.

        **Approach**  
        Lightweight NLP classifier: keyword-based features (25 signals) → Logistic Regression.
        
        **Why it matters**  
        - Cleaner search results  
        - Less manual moderation  
        - Adaptable to other categories
        """
    )
    st.markdown("---")
    st.caption("Note: This demo loads a pre-trained model and does not store any user data.")

# ---------------------------
# Reusable helpers
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

def run_predictions(texts, flag_threshold=0.65):
    raw = pipe.predict(texts)
    labels = to_labels(raw)
    conf = confidences(texts, labels).round(3)
    df = pd.DataFrame({"text": texts, "predicted_label": labels, "confidence": conf})
    df["flag_low_conf"] = df["confidence"] < flag_threshold
    return df

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
    flag_thresh = st.slider("Flag if confidence below", 0.50, 0.95, 0.65, 0.01, key="thr_demo")

    if st.button("Classify", type="primary", key="btn_demo"):
        lines = [ln.strip() for ln in user_text.splitlines() if ln.strip()]
        if not lines:
            st.warning("Please enter at least one line.")
        else:
            df_out = run_predictions(lines, flag_threshold=flag_thresh)
            # Nicely formatted results
            for _, row in df_out.iterrows():
                badge_cls = "badge-ok" if row["predicted_label"] == "computers" else "badge-warn"
                st.markdown(
                    f'<span class="badge {badge_cls}">{row["predicted_label"]}</span> '
                    f'**{row["confidence"]:.2f}** — {row["text"]}',
                    unsafe_allow_html=True,
                )
            st.download_button(
                "Download results (CSV)",
                df_out.to_csv(index=False),
                "predictions.csv",
                key="dl_demo",
            )

# --- Batch tab
with tab_batch:
    st.subheader("Batch Classification (CSV)")
    st.caption("Upload a CSV with a **text** column (or **title** + **description**).")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="uploader")
    # Provide a tiny sample file for convenience
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
        df_out = run_predictions(df_in["text"].tolist(), flag_threshold=0.65)
        st.dataframe(df_out, use_container_width=True)
        st.download_button("Download results (CSV)", df_out.to_csv(index=False), "predictions.csv", key="dl_batch")

# --- Explainability tab
with tab_explain:
    st.subheader("Keyword Signals Detected")
    st.caption("For each listing, we show the keywords (from the 25 training features) found in the text, "
               "and their contribution sign based on the model’s coefficients.")
    txt = st.text_area("Enter one listing per line:", height=160, key="ta_explain")
    if st.button("Analyze keywords", key="btn_explain"):
        lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        if not lines:
            st.warning("Please enter at least one line.")
        else:
            df_pred = run_predictions(lines)
            rows = explain_rows(lines)
            for i, line in enumerate(lines):
                st.markdown(f"**Text:** {line}")
                st.markdown(
                    f'Prediction: <span class="badge badge-ok">{df_pred.loc[i,"predicted_label"]}</span> '
                    f'&nbsp;&middot;&nbsp; Confidence: **{df_pred.loc[i,"confidence"]:.2f}**',
                    unsafe_allow_html=True
                )
                contribs = rows[i]
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
