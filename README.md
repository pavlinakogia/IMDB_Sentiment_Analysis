# IMDB Sentiment Analysis — NLP Classification

An end-to-end NLP project that classifies movie reviews as positive 
or negative using TF-IDF vectorization and machine learning classifiers.

---

## Project Structure

IMDB_Sentiment_Analysis/

├── data.py            # Dataset download & preparation

├── preprocessing.py   # Text cleaning & TF-IDF vectorization

├── model.py           # Model training & evaluation

├── predict.py         # Predict sentiment on custom reviews

├── best_model.pkl     # Saved best model

├── tfidf.pkl          # Saved TF-IDF vectorizer

└── README.md          # This file

---

## What This Project Covers

- Downloading and loading the IMDB dataset (50,000 reviews)
- Text preprocessing: HTML removal, stopword filtering, tokenisation
- TF-IDF vectorization with unigrams and bigrams (10,000 features)
- Training and comparing two classifiers:
  Logistic Regression and Multinomial Naive Bayes
- Evaluation with Accuracy, F1-Score, and Confusion Matrix
- Top predictive words per sentiment class
- Live prediction on custom text input

---

## Results

| Model               | Accuracy | F1-Score |
|---------------------|----------|----------|
| Logistic Regression | 88.6%    | 0.886    |
| Naive Bayes         | 85.4%    | 0.853    |

**Best model:** Logistic Regression (F1 = 0.886)

### Top Predictive Words
- **Positive:** great, excellent, perfect, wonderful, best, amazing
- **Negative:** worst, bad, awful, waste, boring, terrible

### Key Observations
- Balanced dataset (12,500 pos / 12,500 neg) → accuracy is reliable here
- TF-IDF struggles with negation — "not great" misclassified as negative
  because "not" is removed as a stopword and "great" scores as positive
- Short ambiguous reviews (e.g. "not a fan") produce low-confidence 
  predictions (~51%) — honest reflection of model uncertainty
- These limitations motivate the use of contextual models like BERT,
  which understand word meaning in context

---

## Requirements

pip install scikit-learn nltk pandas scipy

---

## How to Run

Run scripts in order:

1. python data.py
2. python preprocessing.py
3. python model.py
4. python predict.py

---

