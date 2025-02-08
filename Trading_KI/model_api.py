from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# KI-Modell laden
model_path = "stock_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

@app.post("/predict")
def predict_stock(data: dict):
    if model is None:
        return {"error": "KI-Modell nicht gefunden!"}
    
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"buy_signal": int(prediction[0])}
