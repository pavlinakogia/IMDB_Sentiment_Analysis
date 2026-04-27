import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ── 1. Φόρτωση δεδομένων ──────────────────────────────────────────────────
X_train = sp.load_npz('X_train.npz')
X_test  = sp.load_npz('X_test.npz')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test  = pd.read_csv('y_test.csv').squeeze()

print(f"X_train: {X_train.shape}")
print(f"Κατανομή y_train:\n{y_train.value_counts()}")

# ── 2. Εκπαίδευση μοντέλων ────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0),
    'Naive Bayes':         MultinomialNB(alpha=0.1)
}

results = {}

for name, model in models.items():
    print(f"\nΕκπαίδευση {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)

    results[name] = {'acc': acc, 'f1': f1, 'model': model, 'pred': y_pred}

    print(f"Accuracy: {acc:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(classification_report(y_test, y_pred,
                                 target_names=['Negative','Positive']))

# ── 3. Σύγκριση μοντέλων ──────────────────────────────────────────────────
names  = list(results.keys())
accs   = [results[n]['acc'] for n in names]
f1s    = [results[n]['f1']  for n in names]

x = range(len(names))
fig, ax = plt.subplots(figsize=(7, 4))
bars1 = ax.bar([i - 0.2 for i in x], accs, 0.4,
                label='Accuracy', color='#3498db')
bars2 = ax.bar([i + 0.2 for i in x], f1s,  0.4,
                label='F1-Score', color='#2ecc71')
ax.set_xticks(list(x))
ax.set_xticklabels(names)
ax.set_ylim(0.8, 1.0)
ax.set_title('Σύγκριση Μοντέλων — IMDB Sentiment')
ax.legend()
for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2,
                           bar.get_height()+0.002,
                           f'{bar.get_height():.3f}',
                           ha='center', fontsize=9)
for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2,
                           bar.get_height()+0.002,
                           f'{bar.get_height():.3f}',
                           ha='center', fontsize=9)
plt.tight_layout()
plt.show()

# ── 4. Confusion Matrix καλύτερου μοντέλου ────────────────────────────────
best_name = max(results, key=lambda x: results[x]['f1'])
best_pred = results[best_name]['pred']

cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Negative','Positive'],
            yticklabels=['Negative','Positive'])
plt.title(f'Confusion Matrix — {best_name}')
plt.ylabel('Πραγματικό')
plt.xlabel('Προβλεπόμενο')
plt.tight_layout()
plt.show()

# ── 5. Top λέξεις ανά sentiment ───────────────────────────────────────────
lr = results['Logistic Regression']['model']
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

feature_names = tfidf.get_feature_names_out()
coefficients  = lr.coef_[0]

top_positive = pd.Series(coefficients, index=feature_names).nlargest(15)
top_negative = pd.Series(coefficients, index=feature_names).nsmallest(15)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
top_positive.plot(kind='barh', ax=axes[0], color='#2ecc71')
axes[0].set_title('Top 15 λέξεις → Positive')
top_negative.abs().sort_values().plot(kind='barh', ax=axes[1], color='#e74c3c')
axes[1].set_title('Top 15 λέξεις → Negative')
plt.tight_layout()
plt.show()

# ── 6. Αποθήκευση καλύτερου μοντέλου ─────────────────────────────────────
best_model = results[best_name]['model']
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"\n Αποθηκεύτηκε το {best_name} ως best_model.pkl")