from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk

app = FastAPI(title="Sentiment Analysis API", version="1.0")

try:
    vectorizer = joblib.load('vectorizer.joblib')
    model = joblib.load('model.joblib')
    stopwords.words('english')
except FileNotFoundError:
    vectorizer = None
    model = None
except LookupError:
    nltk.download('stopwords')


class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: dict

def preprocess_text(text: str) -> str:
    blob = TextBlob(str(text))
    corrected_text = str(blob.correct())

    text = corrected_text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    """Receives text input and returns sentiment prediction."""
    if not model or not vectorizer:
        return {"error": "Model not loaded"}

    cleaned_input = preprocess_text(request.text)
    
    input_vector = vectorizer.transform([cleaned_input])
    prediction = model.predict(input_vector)[0]
    prediction_proba = model.predict_proba(input_vector)
    
    confidence = {
        "negative": round(prediction_proba[0][0] * 100, 2),
        "positive": round(prediction_proba[0][1] * 100, 2)
    }
    
    return SentimentResponse(sentiment=prediction, confidence=confidence)


@app.head("/")

@app.get("/")
def read_root():
    return {"status": "Sentiment Analysis API is running."}
