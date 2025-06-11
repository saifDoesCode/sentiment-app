import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

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
    # Remove user handles (@user)
    text = re.sub(r'@\w+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove non-alphanumeric characters (keeping spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Function to load data, train model, and return vectorizer and model
# Replace your old function with this optimized version
@st.cache_resource
def load_and_train_model():
    # Define the column names for the dataset as it has no header
    column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
    
    # Load the new dataset
    df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', names=column_names)

    # We only need the 'text' and 'target' (sentiment) columns
    data = df[['text', 'target']]

    # --- OPTIMIZATION ---
    # First, sample the data to a smaller size.
    # For even faster startup, you can reduce n from 100000 to 50000.
    data_sample = data.sample(n=100000, random_state=42)
    
    # Second, apply the cleaning function ONLY to the smaller sample.
    # This is much faster than cleaning all 1.6 million rows.
    data_sample['cleaned_text'] = data_sample['text'].apply(preprocess_text)
    
    # Map the target values to readable sentiment labels
    data_sample['sentiment'] = data_sample['target'].map({0: 'negative', 4: 'positive'})

    # Initialize and fit the vectorizer on the cleaned sample
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data_sample['cleaned_text']).toarray()
    y = data_sample['sentiment']

    # Initialize and train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X, y)

    return vectorizer, model
# Load the trained model and vectorizer
with st.spinner('Training AI model... This might take a minute.'):
    vectorizer, model = load_and_train_model()


# --- Streamlit App Interface ---

# --- Streamlit App Interface ---

st.title("💬 Message Sentiment Analyzer")
st.write(
    "Enter a message (like a text, tweet, or chat) to see if its tone is "
    "**Positive** or **Negative**."
)

# User input text area
user_input = st.text_area("Enter the message here:", height=120)

# Analyze button
if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the user's input
        cleaned_input = preprocess_text(user_input)
        input_vector = vectorizer.transform([cleaned_input]) # No .toarray() needed

        # --- NEW CODE: GET PROBABILITIES ---
        # Make the prediction
        prediction = model.predict(input_vector)[0]
        # Get the prediction probabilities
        prediction_proba = model.predict_proba(input_vector)


        # --- DISPLAY THE RESULT AND PERCENTAGES ---
        st.subheader("Analysis Result")
        if prediction == 'positive':
            st.success(f"Sentiment: Positive 😊")
        else:
            st.error(f"Sentiment: Negative 😠")

        # Display the confidence percentages
        st.subheader("Prediction Confidence")
        # The model.classes_ attribute tells us the order of the probabilities
        # For example: ['negative', 'positive']
        # So prediction_proba[0][0] is the probability of 'negative'
        # and prediction_proba[0][1] is the probability of 'positive'
        st.write(f"Negative: {prediction_proba[0][0]*100:.2f}%")
        st.write(f"Positive: {prediction_proba[0][1]*100:.2f}%")

    else:
        st.warning("Please enter a message to analyze.")

st.markdown("---")
st.write("AI model trained on the Sentiment140 dataset.")