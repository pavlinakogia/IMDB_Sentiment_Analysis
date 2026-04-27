from sklearn.datasets import fetch_openml
import pandas as pd

# Το IMDB dataset είναι built-in στο sklearn
print("Φόρτωση dataset...")
from sklearn.datasets import load_files
import urllib.request
import os
import tarfile

# Κατέβασμα IMDB dataset
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filename = "aclImdb_v1.tar.gz"

if not os.path.exists("aclImdb"):
    print("Κατεβάζω dataset... (θα πάρει 1-2 λεπτά)")
    urllib.request.urlretrieve(url, filename)
    print("Αποσυμπίεση...")
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall()
    os.remove(filename)
    print("Έτοιμο!")
else:
    print("Dataset υπάρχει ήδη!")

# Φόρτωση
train_data = load_files("aclImdb/train", categories=['pos','neg'],
                         encoding='utf-8', decode_error='replace')
test_data  = load_files("aclImdb/test",  categories=['pos','neg'],
                         encoding='utf-8', decode_error='replace')

# Μετατροπή σε DataFrame
df_train = pd.DataFrame({'review': train_data.data,
                          'sentiment': train_data.target})
df_test  = pd.DataFrame({'review': test_data.data,
                          'sentiment': test_data.target})

# 0=neg, 1=pos
df_train['sentiment'] = df_train['sentiment'].map({0:'negative', 1:'positive'})
df_test['sentiment']  = df_test['sentiment'].map({0:'negative',  1:'positive'})

print(f"\nTrain: {df_train.shape[0]} κριτικές")
print(f"Test:  {df_test.shape[0]} κριτικές")
print(f"\nΚατανομή train:")
print(df_train['sentiment'].value_counts())

print(f"\nΠαράδειγμα κριτικής:")
print(df_train['review'][0][:300])

# Αποθήκευση
df_train.to_csv('train.csv', index=False)
df_test.to_csv('test.csv',   index=False)
print("\n Αποθηκεύτηκαν ως train.csv και test.csv")