
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

folder = Path(__file__).resolve().parent

df = pd.read_csv(folder / "spam.csv", encoding="latin-1")

if {"label", "message"}.issubset(df.columns):
    X, y = df["message"], df["label"]
elif {"v1", "v2"}.issubset(df.columns):
    X, y = df["v2"], df["v1"]
else:
    raise ValueError("Expected label/message or v1/v2 columns.")

y = y.astype(str).str.lower().str.strip()
y = y.replace({"non-spam": "ham", "nonspam": "ham"})
X = X.fillna("").astype(str)

valid = y.isin(["spam", "ham"]) & (X.str.strip() != "")
X, y = X[valid], y[valid]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

probe = TfidfVectorizer(ngram_range=(1, 2))
k = min(3000, probe.fit_transform(X_train).shape[1])

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("chi2", SelectKBest(chi2, k=k)),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

joblib.dump(model, folder / "spam_model.pkl")

print("Model saved successfully!")
