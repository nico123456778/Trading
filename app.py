import os
import pandas as pd
import pickle
import yfinance as yf
import requests
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from textblob import TextBlob
import ta  # Technische Analyse Bibliothek
import time
import multitasking

# Globale Variable für KI-Bewertungen
global_scores = []
CACHE_FILE = "cached_stock_data.csv"

# 🔥 Maximal 5 gleichzeitige Downloads, um API-Rate-Limits zu vermeiden
multitasking.set_max_threads(5)

def load_cached_data():
    """Lädt gespeicherte Daten, falls vorhanden"""
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE)
    return pd.DataFrame(columns=["Ticker", "Close"])

@multitasking.task
def fetch_data_with_cache(ticker):
    """Lädt Aktien-Daten nur, wenn sie nicht bereits gespeichert wurden"""
    existing_data = load_cached_data()

    if ticker in existing_data["Ticker"].values:
        print(f"⏩ {ticker} bereits gespeichert, überspringe Abruf.")
        return

    try:
        print(f"📊 Lade Daten für {ticker} ...")
        data = yf.download(ticker, period="7d", interval="1d")

        if data.empty or "Close" not in data.columns:
            print(f"⚠️ Keine Daten für {ticker}, überspringe...")
            return
        
        new_data = pd.DataFrame({
            "Ticker": [ticker],
            "Close": [data["Close"].iloc[-1]]
        })
        new_data.to_csv(CACHE_FILE, mode="a", header=not os.path.exists(CACHE_FILE), index=False)
        print(f"✅ {ticker} gespeichert.")

    except Exception as e:
        print(f"❌ Fehler bei {ticker}: {e}")
    
    time.sleep(2)  # 🔥 Wartezeit, um API-Sperren zu vermeiden

# FastAPI App erstellen
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google API Umgebungsvariablen
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Liste der zu analysierenden Aktien und Kryptowährungen
stock_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BTC-USD", "ETH-USD"]  # Kurzfassung
print("Folgende Werte werden analysiert:")
for ticker in stock_list:
    print("-", ticker)
print(f"Gesamtanzahl: {len(stock_list)}")

# KI-Modell laden
model_path = "Trading_KI/models/stock_model.pkl"
if not os.path.exists(model_path):
    print("🚨 Fehler: Die Datei 'stock_model.pkl' existiert nicht!")
    model = None
else:
    try:
        model = joblib.load(model_path)
        print("✅ Modell erfolgreich geladen!")
    except Exception as e:
        print(f"❌ Fehler beim Laden des Modells: {e}")
        model = None

# Funktion zur Sentiment-Analyse
def get_news_sentiment(query):
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={GOOGLE_CSE_ID}&key={GOOGLE_API_KEY}"
        response = requests.get(url).json()
        news = response.get("items", [])[:5]
        sentiment_scores = [TextBlob(article.get("snippet", "")).sentiment.polarity for article in news]
        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
    except Exception:
        return 0.0

# Auswahl der besten Aktie/Krypto
def select_best_asset():
    global global_scores  # WICHTIG: Globale Variable setzen
    scores = []
    
    for ticker in stock_list:
        try:
            print(f"📊 Lade Daten für {ticker} ...")
            data = yf.download(ticker, period="7d", interval="1d")
            if data.empty or "Close" not in data.columns:
                print(f"⚠️ Keine gültigen Daten für {ticker}, überspringe...")
                continue

            print(data.tail())
            df = pd.DataFrame({
                "Close": float(data["Close"].iloc[-1]),
                "RSI": float(ta.momentum.RSIIndicator(data["Close"]).rsi().dropna().values[-1]) if not ta.momentum.RSIIndicator(data["Close"]).rsi().dropna().empty else None,
                "MACD": float(ta.trend.MACD(data["Close"]).macd().dropna().values[-1]) if not ta.trend.MACD(data["Close"]).macd().dropna().empty else None,
            })
            print(f"📊 Berechnete Indikatoren für {ticker}: {df.to_dict(orient='records')}")
            
            prediction = 0
            if model and not df.isnull().values.any():
                prediction = model.predict(df)[0]
                sentiment = get_news_sentiment(ticker)
                final_score = prediction + sentiment
                print(f"🤖 KI-Einschätzung für {ticker}: Prediction={prediction}, Sentiment={sentiment}, Final Score={final_score}")
                scores.append((ticker, final_score))
                global_scores.append((ticker, final_score))
        except Exception as e:
            print(f"❌ Fehler bei {ticker}: {e}")

    best_asset = max(scores, key=lambda x: x[1], default=(None, 0))
    print(f"🏆 Beste Aktie/Krypto: {best_asset[0]} mit Score {best_asset[1]}")
    return best_asset

@app.get("/")
def serve_index():
    try:
        best_asset, score = select_best_asset()
        return {"best_asset": best_asset, "score": score, "full_list": stock_list}
    except Exception as e:
        return {"error": f"Fehler: {e}"}

@app.get("/debug_model")
def debug_model():
    if model is None:
        return {"error": "Modell nicht geladen!"}
    try:
        test_data = pd.DataFrame([[150, 55, 0.5]], columns=["Close", "RSI", "MACD"])
        prediction = int(model.predict(test_data)[0])
        return {"test_prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

if model:
    print("📊 Starte parallelen Datenabruf...")
    for ticker in stock_list:
        fetch_data_with_cache(ticker)
    multitasking.wait_for_tasks()
