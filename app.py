import os
import time
import pandas as pd
import yfinance as yf
import requests
import numpy as np  # Importiere NumPy für NaN-Prüfung
from fastapi import FastAPI
from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import math
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# FastAPI App erstellen
app = FastAPI()

# Statische Dateien bereitstellen
app.mount("/static", StaticFiles(directory="static"), name="static")

# Google API Umgebungsvariablen aus Render laden
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Datenbank einrichten (SQLite für einfaches Deployment)
DATABASE_URL = "sqlite:///./stocks.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Aktienliste
STOCK_LIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "AMD", "BA"]

# Test-Endpunkt zum Überprüfen der Umgebungsvariablen
@app.get("/env-test")
def test_env():
    return {
        "google_api_key": GOOGLE_API_KEY if GOOGLE_API_KEY else "❌ Not found",
        "google_cse_id": GOOGLE_CSE_ID if GOOGLE_CSE_ID else "❌ Not found"
    }

# Google-Suche für relevante Finanznachrichten
@app.get("/search")
def search(q: str):
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return {"error": "Google API Key oder CSE ID fehlt!"}
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": q
    }
    response = requests.get(url, params=params)
    return response.json()

# Funktion zur Berechnung technischer Indikatoren
def calculate_indicators(symbol):
    df = yf.download(symbol, period="6mo", interval="1d")

    if df.empty:
        return None

    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(14).mean() / df["Close"].pct_change().rolling(14).std()))
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()

    latest_data = df.iloc[-1]

    return {
        "symbol": symbol,
        "rsi": latest_data["RSI"],
        "macd": latest_data["MACD"],
        "sma_50": latest_data["SMA_50"],
        "sma_200": latest_data["SMA_200"],
    }

# Funktion zur Auswahl der besten Aktie
def select_best_stock():
    best_stock = None
    best_score = float('-inf')

    for stock in STOCK_LIST:
        indicators = calculate_indicators(stock)
        if not indicators:
            continue

        score = 0
        if indicators["rsi"] is not None and indicators["rsi"].iloc[-1] < 30:
            score += 2
        if indicators["macd"] is not None and indicators["macd"].iloc[-1] > 0:
            score += 1

        if score > best_score:
            best_score = score
            best_stock = indicators

    return best_stock

# Funktion zur Bereinigung ungültiger Werte
def clean_value(value, key):
    if isinstance(value, (float, np.float32, np.float64)):
        if np.isnan(value) or np.isinf(value):
            print(f"⚠ WARNUNG: Ungültiger Wert bei {key}: {value}, wird auf None gesetzt")
            return None
        return round(value, 6)
    return value

import numpy as np

@app.get("/recommendation")
def get_best_stock():
    try:
        best_stock = select_best_stock()  # Wählt die beste Aktie aus
        if not best_stock:
            return {"error": "Keine Empfehlung verfügbar"}

        # Nachrichten auswerten (später mit KI)
        news = [
            {"title": "Marktanalyse: Warum AAPL stark ansteigt", "rating": 9},
            {"title": "Analysten sehen Potenzial für MSFT", "rating": 8},
        ]

        # Bereinige ungültige Werte (NaN oder Inf) für JSON-Kompatibilität
        safe_recommendation = {}
        for key, value in best_stock.items():
            if isinstance(value, (float, np.float32, np.float64)):
                if np.isnan(value) or np.isinf(value):
                    print(f"⚠ WARNUNG: {key} hat ungültigen Wert: {value}")
                    safe_recommendation[key] = None  # Ersetze ungültige Werte mit None
                else:
                    safe_recommendation[key] = round(value, 6)  # Runden für JSON-Sicherheit
            else:
                safe_recommendation[key] = value  # Falls kein Float, direkt übernehmen

        safe_recommendation["news"] = news
        safe_recommendation["history"] = {
            "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "prices": [150, 152, 149]
        }

        print(f"📊 Empfehlung gesendet: {safe_recommendation}")  # Debugging in den Logs
        return safe_recommendation

    except Exception as e:
        print(f"🔥 FEHLER in /recommendation: {str(e)}")  # Fehler in Logs anzeigen
        return {"error": "Internal Server Error", "details": str(e)}


# Startseite auf `index.html` umleiten
@app.get("/")
def read_root():
    return FileResponse("static/index.html")
