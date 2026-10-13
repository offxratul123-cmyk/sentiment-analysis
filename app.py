import re
import streamlit as st
import joblib
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
negations = {"not", "no", "nor", "never", "neither", "nobody", "nothing", "nowhere", "cannot"}
stop_words = stop_words - negations

def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = text.lower().split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("Sentiment Analysis Tool")
st.write("Enter a review below to check whether it is Positive or Negative.")

user_input = st.text_area("Your Review", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        confidence = max(proba) * 100

        if prediction == 1:
            st.success("Prediction: POSITIVE")
        else:
            st.error("Prediction: NEGATIVE")

        st.write(f"Confidence: {confidence:.2f}%")
        st.write(f"Positive score: {proba[1]*100:.2f}%")
        st.write(f"Negative score: {proba[0]*100:.2f}%")