import streamlit as st
import joblib
import re
from pathlib import Path

st.set_page_config(page_title="Spam Email Detection", page_icon="📧", layout="centered")

MODEL_PATH = Path(__file__).parent / "spam_model.joblib"
VECTORIZER_PATH = Path(__file__).parent / "tfidf_vectorizer.joblib"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)

model, vectorizer = load_model()

st.title("📧 Spam Email Detection System")
st.write("Enter an email or message below. The machine-learning model will predict whether it looks like spam or not spam.")

with st.expander("About this project"):
    st.write("""
    This MCA mini project uses Natural Language Processing (NLP), TF-IDF text features,
    and a Naive Bayes classifier. It is a learning demonstration, not a guaranteed
    email-security product. Do not enter private or confidential messages.
    """)

email_text = st.text_area(
    "Email/message text",
    placeholder="Example: Congratulations! You have won a prize. Click here to claim now.",
    height=180
)

if st.button("Check Message", type="primary", use_container_width=True):
    if not email_text.strip():
        st.warning("Please enter some email or message text first.")
    elif model is None or vectorizer is None:
        st.error("Model files are missing. Run `python train_model.py` in the project folder, then restart the app.")
    else:
        features = vectorizer.transform([email_text])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        class_names = list(model.classes_)
        confidence = float(max(probabilities)) * 100

        if prediction == "spam":
            st.error("Prediction: SPAM")
            st.write("This message contains patterns the model learned from spam examples.")
        else:
            st.success("Prediction: NOT SPAM")
            st.write("The model did not identify this message as spam.")
        st.metric("Model confidence", f"{confidence:.1f}%")
        st.caption("Confidence is the model's estimate, not a guarantee that the prediction is correct.")

st.divider()
st.caption("MCA Mini Project | Python • Streamlit • Scikit-learn • NLP")
