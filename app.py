import os
import pandas as pd
import yfinance as yf
import requests
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from textblob import TextBlob
import ta  # Technische Analyse Bibliothek

# FastAPI App erstellen
app = FastAPI()

# CORS aktivieren, falls API von einer externen Website aufgerufen wird
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google API Umgebungsvariablen aus Render laden
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Aktien- und Krypto-Listen
STOCK_LIST = [
    "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META", "NFLX", "MMM", "ABT", "ABBV", "ABMD", "ACN", "ATVI", "ADBE", "AMD", "AAP", "AES", "AFL", "A", "APD", "AKAM", "ALK", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP", "AXP", "AIG", "AMT", "AWK", "AMP", "ABC", "AME", "AMGN", "APH", "ADI", "ANSS", "AON", "APA", "AAPL", "AMAT", "APTV", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK", "AZO", "AVB", "AVY", "BKR", "BALL", "BAC", "BBWI", "BAX", "BDX", "BRK.B", "BBY", "BIO", "TECH", "BIIB", "BLK", "BK", "BA", "BKNG", "BWA", "BXP", "BSX", "BMY", "AVGO", "BR", "BRO", "BF.B", "CHRW", "CDNS", "CZR", "CPB", "COF", "CAH", "KMX", "CCL", "CARR", "CTLT", "CAT", "CBOE", "CBRE", "CDW", "CE", "CNC", "CNP", "CDAY", "CERN", "CF", "CRL", "SCHW", "CHTR", "CVX", "CMG", "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C"]
CRYPTO_LIST = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "DOT-USD", "MATIC-USD", "XLM-USD", "ADA-USD", "DOGE-USD", "LTC-USD", "BCH-USD", "UNI-USD", "LINK-USD", "XMR-USD", "ETC-USD", "VET-USD", "FIL-USD", "TRX-USD", "EOS-USD", "XTZ-USD", "ATOM-USD", "AAVE-USD", "MKR-USD", "ALGO-USD", "DASH-USD", "ZEC-USD"]
ASSET_LIST = STOCK_LIST + CRYPTO_LIST

# KI-Modell laden mit Debugging
model_path = "stock_model.pkl"

if not os.path.exists(model_path):
    print("🚨 Fehler: Die Datei 'stock_model.pkl' existiert nicht im Verzeichnis!")
    model = None
else:
    try:
        model = joblib.load(model_path)
        print("✅ Modell erfolgreich geladen!")
    except Exception as e:
        print(f"❌ Fehler beim Laden des Modells: {e}")
        model = None


# Funktion zur Sentiment-Analyse von Finanznachrichten
def get_news_sentiment(query):
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={GOOGLE_CSE_ID}&key={GOOGLE_API_KEY}"
        response = requests.get(url).json()
        news = response.get("items", [])[:5]
        
        sentiment_scores = []
        for article in news:
            snippet = article.get("snippet", "")
            analysis = TextBlob(snippet)
            sentiment_scores.append(analysis.sentiment.polarity)
        
        if sentiment_scores:
            return sum(sentiment_scores) / len(sentiment_scores)
    except Exception:
        pass
    return 0.0  # Fallback-Sentiment

# Auswahl der besten Aktie/Krypto
def select_best_asset():
    scores = []
    for ticker in ASSET_LIST:
        try:
            data = yf.download(ticker, period="7d", interval="1d")

            print(f"📊 Lade Daten für {ticker} ...")  # Debugging-Log
            if data.empty:
                print(f"⚠ Keine Daten für {ticker} gefunden!")
                continue

        try:
    print(data.tail())  # Zeigt die letzten Zeilen der Daten

    df = pd.DataFrame({
        "close": data["Close"].iloc[-1],
        "RSI": ta.momentum.RSIIndicator(data["Close"]).rsi().dropna().values[-1],
        "MACD": ta.trend.MACD(data["Close"]).macd().dropna().values[-1],
        "SMA50": ta.trend.SMAIndicator(data["Close"], window=50).sma_indicator().dropna().values[-1],
        "SMA200": ta.trend.SMAIndicator(data["Close"], window=200).sma_indicator().dropna().values[-1],
    }, index=[0])

    prediction = model.predict(df)[0] if model else 0
    sentiment = get_news_sentiment(ticker)
    final_score = prediction + sentiment
    scores.append((ticker, final_score))

except Exception as e:
    print(f"❌ Fehler bei {ticker}: {e}")
    continue




    if scores:
        return max(scores, key=lambda x: x[1], default=(None, 0.0))
    return None, 0.0


# API-Route für die empfohlene Aktie/Krypto
from fastapi.responses import FileResponse
import os

from fastapi.responses import FileResponse
from fastapi.requests import Request

@app.get("/")
def serve_index(request: Request):
    if "text/html" in request.headers.get("accept", ""):
      return FileResponse("static/index.html")

    else:
        best_asset, score = select_best_asset()
        return {"best_asset": best_asset, "score": score, "full_list": ASSET_LIST}

@app.get("/debug_model")
def debug_model():
    if model is None:
        return {"error": "Modell wurde nicht geladen!"}
    
    try:
        test_data = pd.DataFrame([[150, 55, 0.5, 200, 210]], columns=["close", "RSI", "MACD", "SMA50", "SMA200"])
        prediction = int(model.predict(test_data)[0])
        return {"test_prediction": prediction}
    except Exception as e:
        return {"error": str(e)}
