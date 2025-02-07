from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# 📥 Modell laden
model = joblib.load("models/stock_model.pkl")

@app.post("/predict")
def predict_stock(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"buy_signal": int(prediction[0])}
