import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import requests
import zipfile
import io
import joblib 

print("Downloading dataset...")
DATA_URL = 'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'
r = requests.get(DATA_URL)
z = zipfile.ZipFile(io.BytesIO(r.content))

column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
df = pd.read_csv(z.open('training.1600000.processed.noemoticon.csv'), encoding='latin-1', header=None, names=column_names)
print("Dataset downloaded and loaded.")

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

print("Preprocessing a large sample of data (50,000 rows)...")
data = df[['text', 'target']]
data = df[df['target'] != 2] # Remove neutral tweets if any
data_sample = data.sample(n=100000, random_state=42)
data_sample['cleaned_text'] = data_sample['text'].apply(preprocess_text)
data_sample['sentiment'] = data_sample['target'].map({0: 'negative', 4: 'positive'})
print("Preprocessing complete.")

print("Training the model...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data_sample['cleaned_text'])
y = data_sample['sentiment']
model = MultinomialNB()
model.fit(X, y)
print("Model training complete.")

print("Saving model and vectorizer to files...")
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(model, 'model.joblib')
print("Files saved successfully: vectorizer.joblib, model.joblib")