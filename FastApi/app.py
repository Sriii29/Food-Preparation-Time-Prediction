from fastapi import FastAPI
import joblib
import pandas as pd
from scipy.sparse import hstack, csr_matrix

app = FastAPI()

vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("scaler.pkl")
model_p50 = joblib.load("model_p50.pkl")
model_p90 = joblib.load("model_p90.pkl")

@app.post("/predict")
def predict(data: dict):

    text = vectorizer.transform([data["items"]])

    numeric_df = pd.DataFrame([{
        "hour": data["hour"],
        "rain": data["rain"],
        "event": data["event"],
        "kitchen_load": data["kitchen_load"]
    }])

    numeric_scaled = scaler.transform(numeric_df)

    X = hstack([text, csr_matrix(numeric_scaled)])

    p50 = model_p50.predict(X)[0]
    p90 = model_p90.predict(X)[0]

    return {
        "p50_ready_time": round(float(p50),1),
        "p90_ready_time": round(float(p90),1)
    }
