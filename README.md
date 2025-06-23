📰 Fake News Detection App

A lightweight web application that classifies news as Fake ⚠️ or Real 📰 using Natural Language Processing (NLP) and Machine Learning.

🚀 Project Overview

This project demonstrates a complete pipeline from:

Data ingestion (real + fake news),

Text preprocessing,

Model training using Logistic Regression,

To deploy via a Flask web app.

Simple, fast, and effective — great for educational or lightweight production use.

🧠 Model Architecture
Preprocessing: Basic cleaning (regex), lowercase conversion

Vectorization: TF-IDF with max_features=1000

Model: Logistic Regression (liblinear solver)

Evaluation: Train/Test split with accuracy score

Deployment: Flask web interface with prediction & confidence display

