import pandas as pd
import re
import os
import nltk
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (first time only)
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

fake['label'] = 0
real['label'] = 1

df = pd.concat([fake, real], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)
df['content'] = df['title'] + " " + df['text']

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.lower().split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_content'] = df['content'].apply(clean)

# Feature extraction
X = df['clean_content']
y = df['label']

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# Save model and vectorizer
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl', compress=9)
joblib.dump(vectorizer, 'model/vectorizer.pkl', compress=9)

print("✅ Model and vectorizer saved successfully.")
