import streamlit as st
import pickle
import re
import os

# Δική μας λίστα stopwords — χωρίς NLTK!
STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','he','him','his','himself','she','her','hers','herself','it',
    'its','itself','they','them','their','theirs','themselves','what','which',
    'who','whom','this','that','these','those','am','is','are','was','were',
    'be','been','being','have','has','had','having','do','does','did','doing',
    'a','an','the','and','but','if','or','because','as','until','while','of',
    'at','by','for','with','about','against','between','into','through',
    'during','before','after','above','below','to','from','up','down','in',
    'out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more',
    'most','other','some','such','no','nor','not','only','own','same','so',
    'than','too','very','s','t','can','will','just','don','should','now',
    'd','ll','m','o','re','ve','y','ain','aren','couldn','didn','doesn',
    'hadn','hasn','haven','isn','ma','mightn','mustn','needn','shan',
    'shouldn','wasn','weren','won','wouldn'
}

# ── Φόρτωση μοντέλου & vectorizer ────────────────────────────────────────
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(base_dir, 'tfidf.pkl'), 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_model()

# ── Καθαρισμός κειμένου ───────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)

# ── UI ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Γράψε μια κριτική ταινίας και το μοντέλο θα αναλύσει αν είναι θετική ή αρνητική.")
st.markdown("---")

review = st.text_area(" Γράψε την κριτική σου (στα αγγλικά):",
                       placeholder="e.g. This movie was absolutely amazing...",
                       height=150)

if st.button(" Ανάλυση Συναισθήματος"):
    if review.strip() == "":
        st.warning("Παρακαλώ γράψε μια κριτική πρώτα!")
    else:
        cleaned    = clean_text(review)
        vector     = tfidf.transform([cleaned])
        pred       = model.predict(vector)[0]
        prob       = model.predict_proba(vector)[0]
        confidence = max(prob) * 100

        st.markdown("---")
        if pred == 1:
            st.success(f" POSITIVE — {confidence:.1f}% confidence")
        else:
            st.error(f" NEGATIVE — {confidence:.1f}% confidence")

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