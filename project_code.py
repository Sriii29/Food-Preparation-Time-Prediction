#NLP Embeddings for Ordered Items
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv("food_prep_dataset.csv")

vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(df["items"])

import joblib
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Embeddings created.")
#Feature Engineering
import pandas as pd
import joblib
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("food_prep_dataset.csv")

vectorizer = joblib.load("tfidf_vectorizer.pkl")
X_text = vectorizer.transform(df["items"])

numeric_features = df[["hour", "rain", "event", "kitchen_load"]]

scaler = StandardScaler()
X_numeric = scaler.fit_transform(numeric_features)

joblib.dump(scaler, "scaler.pkl")

from scipy.sparse import csr_matrix
X_combined = hstack([X_text, csr_matrix(X_numeric)])

y = df["prep_time"]

joblib.dump((X_combined, y), "train_data.pkl")

print("Features ready.")

#Quantile Regression (P50 & P90)
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

X, y = joblib.load("train_data.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# P50 model
model_p50 = GradientBoostingRegressor(loss="quantile", alpha=0.5)
model_p50.fit(X_train, y_train)

# P90 model
model_p90 = GradientBoostingRegressor(loss="quantile", alpha=0.9)
model_p90.fit(X_train, y_train)

joblib.dump(model_p50, "model_p50.pkl")
joblib.dump(model_p90, "model_p90.pkl")

print("Models trained.")