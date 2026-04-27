import nltk
import os

# Κατέβασμα NLTK δεδομένων πριν από οτιδήποτε άλλο
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

# ── Φόρτωση μοντέλου & vectorizer ────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('IMDB_Sentiment_Analysis/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('IMDB_Sentiment_Analysis/tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_model()
stop_words = set(stopwords.words('english'))

# ── Καθαρισμός κειμένου ───────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

# ── UI ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Γράψε μια κριτική ταινίας και το μοντέλο θα αναλύσει αν είναι θετική ή αρνητική.")
st.markdown("---")

review = st.text_area(" Γράψε την κριτική σου (στα αγγλικά):",
                       placeholder="e.g. This movie was absolutely amazing...",
                       height=150)

if st.button("🔍 Ανάλυση Συναισθήματος"):
    if review.strip() == "":
        st.warning("Παρακαλώ γράψε μια κριτική πρώτα!")
    else:
        cleaned = clean_text(review)
        vector  = tfidf.transform([cleaned])
        pred    = model.predict(vector)[0]
        prob    = model.predict_proba(vector)[0]
        confidence = max(prob) * 100

        st.markdown("---")
        if pred == 1:
            st.success(f" POSITIVE — {confidence:.1f}% confidence")
        else:
            st.error(f" NEGATIVE — {confidence:.1f}% confidence")

        # Progress bar
        st.markdown("**Κατανομή πιθανότητας:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Negative", f"{prob[0]*100:.1f}%")
            st.progress(float(prob[0]))
        with col2:
            st.metric("Positive", f"{prob[1]*100:.1f}%")
            st.progress(float(prob[1]))

st.markdown("---")
st.markdown("*Model: Logistic Regression + TF-IDF | Dataset: IMDB 50k reviews | Accuracy: 88.6%*")