import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# ── 1. Φόρτωση ────────────────────────────────────────────────────────────
df_train = pd.read_csv('train.csv')
df_test  = pd.read_csv('test.csv')

print("=== Πρώτη ματιά ===")
print(df_train.head(3))
print(f"\nΚατανομή sentiment:")
print(df_train['sentiment'].value_counts())

# ── 2. Καθαρισμός κειμένου ────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Αφαίρεση HTML tags (π.χ. <br />)
    text = re.sub(r'<.*?>', ' ', text)
    # Κράτα μόνο γράμματα
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Μετατροπή σε πεζά
    text = text.lower()
    # Tokenisation & αφαίρεση stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

print("\nΚαθαρισμός κειμένου...")
df_train['clean_review'] = df_train['review'].apply(clean_text)
df_test['clean_review']  = df_test['review'].apply(clean_text)

print("Πριν:", df_train['review'][0][:150])
print("Μετά:", df_train['clean_review'][0][:150])

# ── 3. TF-IDF Vectorization ───────────────────────────────────────────────
print("\nΥπολογισμός TF-IDF...")
tfidf = TfidfVectorizer(max_features=10000,  # κρατάμε τις 10k πιο σημαντικές λέξεις
                         ngram_range=(1, 2),  # unigrams & bigrams
                         min_df=5)            # λέξη πρέπει να εμφανίζεται σε 5+ κριτικές

X_train = tfidf.fit_transform(df_train['clean_review'])
X_test  = tfidf.transform(df_test['clean_review'])

y_train = (df_train['sentiment'] == 'positive').astype(int)
y_test  = (df_test['sentiment']  == 'positive').astype(int)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")

# ── 4. Αποθήκευση ─────────────────────────────────────────────────────────
import scipy.sparse as sp
sp.save_npz('X_train.npz', X_train)
sp.save_npz('X_test.npz',  X_test)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv',   index=False)

# Αποθήκευση TF-IDF vectorizer για αργότερα
with open('../tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("\nΑποθηκεύτηκαν X_train, X_test, y_train, y_test, tfidf.pkl")