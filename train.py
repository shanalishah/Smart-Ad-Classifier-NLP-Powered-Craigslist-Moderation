"""
Train TF-IDF + Logistic Regression classifier.

Prefers a human-labeled file:
  data/labeled_and_flagged_with_human_check.csv  (columns: text, human_label)
Fallback (if you don't have human labels yet):
  data/combined_data.csv  (uses weak labels from scrape: 'label' column)

Outputs:
  pipeline_lr_tfidf.joblib
  data/clean_dedup_labeled.csv  (created if the human-labeled file is present)
"""

from pathlib import Path
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

DATA = Path("data")
HUMAN = DATA / "labeled_and_flagged_with_human_check.csv"
COMBINED = DATA / "combined_data.csv"
CLEAN = DATA / "clean_dedup_labeled.csv"
MODEL = Path("pipeline_lr_tfidf.joblib")

def load_labeled() -> pd.DataFrame:
    if HUMAN.exists():
        df = pd.read_csv(HUMAN)
        # Normalize & clean
        df["human_label"] = df["human_label"].replace({"computer": "computers"})
        df = df[df["human_label"].isin(["computers","computer_parts"])].copy()
        # If text is missing but title/description exist, combine
        if "text" not in df.columns and {"title","description"}.issubset(df.columns):
            df["text"] = (df["title"].fillna("") + " " + df["description"].fillna("")).astype(str)
        df["text"] = df["text"].astype(str)
        df = df[df["text"].str.strip().ne("")]
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        df[["text","human_label"]].to_csv(CLEAN, index=False)
        print(f"[train] Saved cleaned labels -> {CLEAN} ({len(df)} rows)")
        return df[["text","human_label"]].rename(columns={"human_label":"label"})
    elif COMBINED.exists():
        df = pd.read_csv(COMBINED)
        df["text"] = (df.get("text") or (df["title"].fillna("") + " " + df["description"].fillna(""))).astype(str)
        df = df[df["text"].str.strip().ne("")]
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        df = df.rename(columns={"label":"label"})
        print("[train] Using weak labels from combined_data.csv (no human_label found)")
        return df[["text","label"]]
    else:
        raise FileNotFoundError("Provide data/labeled_and_flagged_with_human_check.csv or data/combined_data.csv")

def main():
    df = load_labeled()
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=3000)),
        ("clf", LogisticRegression(max_iter=200, solver="liblinear"))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"[metrics] Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred, labels=["computers","computer_parts"]))

    joblib.dump(pipe, MODEL)
    print(f"[train] Saved model -> {MODEL}")

if __name__ == "__main__":
    main()
