import pickle

# Φόρτωση μοντέλου και vectorizer
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

def predict_sentiment(review):
    cleaned  = clean_text(review)
    vector   = tfidf.transform([cleaned])
    pred     = model.predict(vector)[0]
    prob     = model.predict_proba(vector)[0]
    label    = "POSITIVE" if pred == 1 else "NEGATIVE"
    confidence = max(prob) * 100
    print(f"\nΚριτική: {review[:80]}...")
    print(f"Αποτέλεσμα: {label} ({confidence:.1f}% confidence)")

# Δοκίμασε διάφορες κριτικές
predict_sentiment("This movie was absolutely amazing! The acting was superb.")
predict_sentiment("Terrible film, complete waste of time. Boring and predictable.")
predict_sentiment("It was okay, not great but not bad either.")

# Γράψε τη δική σου!
my_review = input("\nΓράψε τη δική σου κριτική (στα αγγλικά): ")
predict_sentiment(my_review)