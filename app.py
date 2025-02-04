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
    print(f"🔍 Berechne Indikatoren für: {symbol}")  # Debugging

    df = yf.download(symbol, period="6mo", interval="1d")

    if df.empty:
        print(f"⚠ Keine Daten für {symbol} gefunden!")
        return None

    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(14).mean() / df["Close"].pct_change().rolling(14).std()))
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()

    latest_data = df.iloc[-1]

    def safe_float(value, key):
        """ Konvertiert Werte sicher in Float, sonst None """
        if isinstance(value, pd.Series):
            print(f"⚠ WARNUNG: {key} ist eine Pandas-Serie! Extrahiere letzten Wert.")
            value = value.iloc[-1]
        if isinstance(value, (float, int)):
            return float(value)
        print(f"❌ FEHLER: {key} konnte nicht in Float umgewandelt werden! Setze auf None.")
        return None

    result = {
        "symbol": symbol,
        "rsi": safe_float(latest_data["RSI"], "RSI"),
        "macd": safe_float(latest_data["MACD"], "MACD"),
        "sma_50": safe_float(latest_data["SMA_50"], "SMA_50"),
        "sma_200": safe_float(latest_data["SMA_200"], "SMA_200"),
    }

    print(f"✅ Indikatoren für {symbol}: {result}")  # Debugging
    return result




# Funktion zur Auswahl der besten Aktie



def select_best_stock():
    print("🔍 Starte Auswahl der besten Aktie...")  # Debugging

    best_stock = None
    best_score = float('-inf')

    for stock in STOCK_LIST:
        indicators = calculate_indicators(stock)
        if not indicators:
            print(f"⚠ Keine Indikatoren für {stock}, überspringe...")
            continue

        print(f"📊 Prüfe Aktie {stock}: {indicators}")  # Debugging

        score = 0

        rsi_value = indicators.get("rsi")
        macd_value = indicators.get("macd")

        if isinstance(rsi_value, pd.Series):
            print(f"⚠ WARNUNG: RSI für {stock} ist eine Serie! Extrahiere letzten Wert.")
            rsi_value = rsi_value.iloc[-1]
        if isinstance(macd_value, pd.Series):
            print(f"⚠ WARNUNG: MACD für {stock} ist eine Serie! Extrahiere letzten Wert.")
            macd_value = macd_value.iloc[-1]

        rsi_value = float(rsi_value) if isinstance(rsi_value, (float, int)) else None
        macd_value = float(macd_value) if isinstance(macd_value, (float, int)) else None

        if rsi_value is not None and rsi_value < 30:
            score += 2
        if macd_value is not None and macd_value > 0:
            score += 1

        print(f"📈 Score für {stock}: {score}")

        if score > best_score:
            best_score = score
            best_stock = indicators

    print(f"🏆 Beste Aktie: {best_stock}")  # Debugging
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

def clean_json_data(data):
    """ Rekursive Funktion zur Bereinigung von JSON-Daten (NaN, Inf ersetzen). """
    if isinstance(data, dict):
        return {key: clean_json_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_json_data(item) for item in data]
    elif isinstance(data, (float, np.float32, np.float64)):
        if np.isnan(data) or np.isinf(data):
            print(f"⚠ WARNUNG: Ungültiger Wert erkannt: {data}")  # Debugging
            return None  # Ersetze ungültige Werte
        return round(data, 6)  # Runden für JSON-Sicherheit
    else:
        return data  # Andere Werte bleiben unverändert

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

        recommendation = {
            "symbol": best_stock["symbol"],
            "rsi": best_stock["rsi"],
            "macd": best_stock["macd"],
            "sma_50": best_stock["sma_50"],
            "sma_200": best_stock["sma_200"],
            "history": {
                "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "prices": [150, 152, 149]
            },
            "news": news
        }

        # Bereinigung der Daten vor dem JSON-Response
        safe_recommendation = clean_json_data(recommendation)

        print(f"📊 Empfehlung gesendet: {safe_recommendation}")  # Debugging in den Logs
        return safe_recommendation

    except Exception as e:
        print(f"🔥 FEHLER in /recommendation: {str(e)}")  # Fehler in Logs anzeigen
        return {"error": "Internal Server Error", "details": str(e)}


# Startseite auf `index.html` umleiten
@app.get("/")
def read_root():
    return FileResponse("static/index.html")
