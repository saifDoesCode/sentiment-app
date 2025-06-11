import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
import joblib # Import joblib to load our models
from textblob import TextBlob

# --- Page Configuration ---
st.set_page_config(page_title="Message Sentiment Analyzer", page_icon="💬")

# --- Model Loading ---

# Load the pre-trained vectorizer and model
# This is much faster and uses less memory than training from scratch
@st.cache_resource
def load_model_and_vectorizer():
    vectorizer = joblib.load('vectorizer.joblib')
    model = joblib.load('model.joblib')
    return vectorizer, model

def preprocess_text(text):
    # --- NEW STEP: SPELL CORRECTION ---
    # Convert the text to a TextBlob object
    blob = TextBlob(str(text))
    # Correct the spelling
    corrected_text = str(blob.correct())

    # --- The rest of the function is the same ---
    text = corrected_text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Load the models when the app starts
vectorizer, model = load_model_and_vectorizer()


# --- Streamlit App Interface ---
st.title("💬 Message Sentiment Analyzer")
st.write(
    "Enter a message (like a text, tweet, or chat) to see if its tone is "
    "**Positive** or **Negative**."
)

user_input = st.text_area("Enter the message here:", height=120)

if st.button("Analyze Sentiment"):
    if user_input:
        cleaned_input = preprocess_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vector)[0]
        prediction_proba = model.predict_proba(input_vector)

        st.subheader("Analysis Result")
        if prediction == 'positive':
            st.success(f"Sentiment: Positive 😊")
        else:
            st.error(f"Sentiment: Negative 😠")

        st.subheader("Prediction Confidence")
        st.write(f"Negative: {prediction_proba[0][0]*100:.2f}%")
        st.write(f"Positive: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.warning("Please enter a message to analyze.")

st.markdown("---")
st.write("Developed by Saif.")
st.write("AI model is trained on the Sentiment140 dataset. This dataset includes **1.6 Million rows of data.**")