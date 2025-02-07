import os
import time
import pandas as pd
import yfinance as yf
import requests
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
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

# 📥 Aktien- und Krypto-Listen
STOCK_LIST = [
    "AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META", "NFLX",
    "MMM", "ABT", "ABBV", "ABMD", "ACN", "ATVI", "ADBE", "AMD", "AAP", "AES", "AFL", "A", "APD", "AKAM", "ALK", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE", "AAL", "AEP", "AXP", "AIG", "AMT", "AWK", "AMP", "ABC", "AME", "AMGN", "APH", "ADI", "ANSS", "AON", "APA", "AAPL", "AMAT", "APTV", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK", "AZO", "AVB", "AVY", "BKR", "BALL", "BAC", "BBWI", "BAX", "BDX", "BRK.B", "BBY", "BIO", "TECH", "BIIB", "BLK", "BK", "BA", "BKNG", "BWA", "BXP", "BSX", "BMY", "AVGO", "BR", "BRO", "BF.B", "CHRW", "CDNS", "CZR", "CPB", "COF", "CAH", "KMX", "CCL", "CARR", "CTLT", "CAT", "CBOE", "CBRE", "CDW", "CE", "CNC", "CNP", "CDAY", "CERN", "CF", "CRL", "SCHW", "CHTR", "CVX", "CMG", "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C"
]
CRYPTO_LIST = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "DOT-USD", "MATIC-USD", "XLM-USD", "ADA-USD", "DOGE-USD", "LTC-USD", "BCH-USD", "UNI-USD", "LINK-USD", "XMR-USD", "ETC-USD", "VET-USD", "FIL-USD", "TRX-USD", "EOS-USD", "XTZ-USD", "ATOM-USD", "AAVE-USD", "MKR-USD", "ALGO-USD", "DASH-USD", "ZEC-USD"
]
ASSET_LIST = STOCK_LIST + CRYPTO_LIST  # Enthält alle trainierten Aktien & Kryptos

# Makroökonomische Faktoren abrufen
def get_interest_rate():
    try:
        fed_funds = yf.Ticker("^IRX")
        return fed_funds.history(period="1mo")["Close"].iloc[-1]
    except Exception:
        return 0.0  # Fallback-Wert, falls API nicht erreichbar ist

def get_inflation():
    try:
        cpi = yf.Ticker("^CPI")
        return cpi.history(period="1mo")["Close"].iloc[-1]
    except Exception:
        return 0.0  # Fallback-Wert

inflation = get_inflation()
interest_rate = get_interest_rate()

# KI-Modell laden (Pfad angepasst für Render-Deployment)
model_path = os.path.join(os.getcwd(), "Trading-main", "Trading_KI", "models", "stock_model.pkl")
model = joblib.load(model_path)

# 📊 Funktion zur Sentiment-Analyse von Finanznachrichten
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

# 📈 Funktion zur Auswahl der besten Aktie/Krypto
def select_best_asset():
    try:
        sp500 = yf.download("^GSPC", period="5y", interval="1d")["Close"].squeeze()
    except Exception:
        sp500 = pd.Series([1] * 365 * 5)  # Fallback-Werte für S&P 500
    
    scores = []
    for ticker in ASSET_LIST:
        try:
            data = yf.download(ticker, period="7d", interval="1d")
            if data.empty:
                continue
            
            latest_data = data.iloc[-1]
            df = pd.DataFrame([{ 
                "close": latest_data.get("Close", np.nan),
                "volume": latest_data.get("Volume", np.nan),
                "RSI": ta.momentum.RSIIndicator(data["Close"]).rsi().iloc[-1] if "Close" in data else np.nan,
                "MACD": ta.trend.MACD(data["Close"]).macd().iloc[-1] if "Close" in data else np.nan,
                "SMA50": ta.trend.SMAIndicator(data["Close"], window=50).sma_indicator().iloc[-1] if "Close" in data else np.nan,
                "SMA200": ta.trend.SMAIndicator(data["Close"], window=200).sma_indicator().iloc[-1] if "Close" in data else np.nan,
                "ATR": ta.volatility.AverageTrueRange(data["High"], data["Low"], data["Close"]).average_true_range().iloc[-1] if "High" in data else np.nan,
                "BB_Upper": ta.volatility.BollingerBands(data["Close"]).bollinger_hband().iloc[-1] if "Close" in data else np.nan,
                "BB_Lower": ta.volatility.BollingerBands(data["Close"]).bollinger_lband().iloc[-1] if "Close" in data else np.nan,
                "Volume_Trend": latest_data["Volume"] / data["Volume"].rolling(10).mean().iloc[-1] if "Volume" in data else np.nan,
                "Relative_Strength": latest_data["Close"] / sp500.ffill().reindex(data.index, method="nearest").iloc[-1] if "Close" in data else np.nan,
                "Inflation": inflation,
                "Interest_Rate": interest_rate
            }])
            
            df.fillna(0, inplace=True)  # NaN-Werte vermeiden
            prediction = model.predict(df)[0]
            sentiment = get_news_sentiment(ticker)
            final_score = prediction + sentiment
            scores.append((ticker, final_score))
        except Exception:
            continue
    
    if scores:
        return max(scores, key=lambda x: x[1], default=(None, 0.0))
    return None, 0.0

# 📢 API-Route zur Startseite
@app.get("/")
def get_recommendation():
    best_asset, score = select_best_asset()
    return {"best_asset": best_asset, "score": score}
