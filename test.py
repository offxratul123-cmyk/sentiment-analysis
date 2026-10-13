import joblib
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
negations = {"not", "no", "nor", "never", "neither", "nobody", "nothing", "nowhere", "cannot"}
stop_words = stop_words - negations

def clean_text(text):
    tokens = text.lower().split()
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(tokens)

model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

tests = [
    "This movie was absolutely fantastic. The acting was brilliant and the story kept me engaged from start to finish.",
    "Worst movie I have ever seen. Boring plot, terrible acting, complete waste of time."
]

for t in tests:
    cleaned = clean_text(t)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    print("Text:", t[:60])
    print("Prediction:", pred, "| pos:", round(proba[1]*100,2), "| neg:", round(proba[0]*100,2))
    print("-" * 60)