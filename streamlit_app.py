import streamlit as st
import pandas as pd
import joblib
import os

# ------------------------
# PAGE CONFIGURATION
# ------------------------
st.set_page_config(
    page_title="Smart Ad Classifier – Craigslist Listings",
    layout="wide"
)

# ------------------------
# LOAD PRETRAINED MODELS
# ------------------------
VEC_PATH = "models/tfidf_vectorizer.pkl"
MODEL_PATH = "models/final_logistic_model.pkl"

@st.cache_resource
def load_models():
    if os.path.exists(VEC_PATH) and os.path.exists(MODEL_PATH):
        vectorizer = joblib.load(VEC_PATH)
        model = joblib.load(MODEL_PATH)
        return vectorizer, model
    else:
        st.error("Model files not found. Please ensure both .pkl files are in the data/ directory.")
        st.stop()

vectorizer, model = load_models()

# ------------------------
# APP TITLE
# ------------------------
st.title("🛠 Smart Ad Classifier – Craigslist Computer Listings")
st.markdown(
    """
    This tool classifies Craigslist listings into **"Computer"** or **"Computer Part"** using a 
    machine learning model trained on real scraped data.
    """
)

# ------------------------
# TABS
# ------------------------
tab1, tab2 = st.tabs(["💬 Quick Test", "📂 Bulk Classification (CSV)"])

# ------------------------
# TAB 1 – QUICK TEST
# ------------------------
with tab1:
    st.subheader("Quick Test – Classify a Single Listing")
    st.markdown("Enter a product title or short description to see how the model classifies it.")

    user_input = st.text_area("Listing text:", placeholder="Example: Apple MacBook Pro 16-inch, 16GB RAM, 512GB SSD")

    if st.button("Classify Listing", key="btn_single_classify"):
        if user_input.strip():
            X = vectorizer.transform([user_input])
            pred = model.predict(X)[0]
            proba = model.predict_proba(X).max()

            st.success(f"**Prediction:** {pred}")
            st.info(f"Confidence: {proba:.2%}")
        else:
            st.warning("Please enter some text before classifying.")

# ------------------------
# TAB 2 – BULK CSV UPLOAD
# ------------------------
with tab2:
    st.subheader("Bulk Classification – Upload a CSV File")
    st.markdown("Upload a CSV containing a column named **`text`** with listing descriptions.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if "text" not in df.columns:
                st.error("CSV must contain a column named 'text'.")
            else:
                X = vectorizer.transform(df["text"])
                preds = model.predict(X)
                probs = model.predict_proba(X).max(axis=1)

                df_out = df.copy()
                df_out["predicted_label"] = preds
                df_out["confidence"] = probs.round(3)

                st.dataframe(df_out, use_container_width=True)

                csv_data = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Predictions",
                    data=csv_data,
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
st.caption("This demo uses a TF-IDF vectorizer with Logistic Regression, trained on labeled Craigslist data.")
