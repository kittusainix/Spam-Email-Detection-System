from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "spam.csv"
MODEL_PATH = ROOT / "spam_model.joblib"
VECTORIZER_PATH = ROOT / "tfidf_vectorizer.joblib"

def main():
    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=["label", "text"])
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["label"].isin(["spam", "ham"])]
    data = data[data["text"] != ""]
    data["label"] = data["label"].replace({"ham": "not_spam"})

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.25, random_state=42, stratify=data["label"]
    )

    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=5000
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)

    print("\\n--- Model Evaluation ---")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix (label order shown by sklearn):")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"\\nSaved model: {MODEL_PATH.name}")
    print(f"Saved vectorizer: {VECTORIZER_PATH.name}")
    print("Now run: streamlit run app.py")

if __name__ == "__main__":
    main()
