import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import requests # We need to import the requests library
import zipfile # We need to import the zipfile library
import io      # We need to import the io library


# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Message Sentiment Analyzer", page_icon="💬")


# --- Preprocessing and Model Training ---

# Ensure the stopwords are downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Text preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Function to load data, train model, and return vectorizer and model
# --- IMPORTANT CHANGES ARE IN THIS FUNCTION ---
@st.cache_data
def load_and_train_model():
    # URL pointing to the ZIP file
    DATA_URL = 'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'
    
    # --- NEW CODE TO HANDLE THE ZIP FILE ---
    # Download the ZIP file from the URL
    r = requests.get(DATA_URL)
    # Create a file-like object in memory from the content
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    # Define the column names for the dataset
    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    # Extract and read the correct training CSV file from the ZIP archive
    df = pd.read_csv(
        z.open('training.1600000.processed.noemoticon.csv'), # Specify the filename inside the ZIP
        encoding='latin-1',
        header=None,
        names=column_names
    )

    # The rest of your function remains the same
    data = df[['text', 'target']]
    data_sample = data.sample(n=3000, random_state=42)
    data_sample['cleaned_text'] = data_sample['text'].apply(preprocess_text)
    data_sample['sentiment'] = data_sample['target'].map({0: 'negative', 4: 'positive'})

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data_sample['cleaned_text'])
    y = data_sample['sentiment']

    model = MultinomialNB()
    model.fit(X, y)

    return vectorizer, model

# Load the trained model and vectorizer
with st.spinner('Downloading data and training model... This may take a minute on first run.'):
    vectorizer, model = load_and_train_model()


# --- Streamlit App Interface (No changes here) ---

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
st.markdown("Developed by Saif.")
st.markdown("AI model is trained on the Sentiment140 dataset. **The Dataset includes 1.6 Million tweets!**")