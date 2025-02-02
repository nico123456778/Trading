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

# Datenbank einrichten (SQLite für einfaches Deployment)
DATABASE_URL = "sqlite:///./stocks.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Aktienliste
STOCK_LIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "AMD", "BA"]

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

    print("🔍 Starte Auswahl der besten Aktie...")

    for stock in STOCK_LIST:
        indicators = calculate_indicators(stock)
        if not indicators:
            print(f"❌ Keine Daten für {stock}")
            continue

        for key in ["rsi", "macd", "sma_50", "sma_200"]:
            if isinstance(indicators[key], pd.Series):
                indicators[key] = indicators[key].iloc[-1]

        print(f"📊 Daten für {stock}: {indicators}")

        score = 0
        if indicators["rsi"] is not None and indicators["rsi"] < 30:  
            score += 2
        if indicators["macd"] is not None and indicators["macd"] > 0:  
            score += 1

        print(f"⚖ Bewertung für {stock}: Score {score}")

        if score > best_score:
            best_score = score
            best_stock = indicators

    if best_stock:
        print(f"✅ Beste Aktie: {best_stock['symbol']} mit Score {best_score}")
    else:
        print("❌ Keine Aktie gefunden!")

    return best_stock

# API-Endpunkt für die beste Aktie
@app.get("/recommendation")
def get_best_stock():
    try:
        best_stock = select_best_stock()
        if not best_stock:
            print("❌ Keine Aktie gefunden")
            return {"error": "Keine Empfehlung verfügbar"}

        print(f"✅ Beste Aktie ausgewählt: {best_stock}")

        news = [
            {"title": "Marktanalyse: Warum AAPL stark ansteigt", "rating": 9},
            {"title": "Analysten sehen Potenzial für MSFT", "rating": 8},
        ]

        def clean_value(value, key):
            if isinstance(value, (float, np.float32, np.float64)):
                if np.isnan(value) or np.isinf(value):
                    print(f"⚠ WARNUNG: Ungültiger Wert bei {key}: {value}, wird auf None gesetzt")
                    return None
                return round(value, 6)
            return value

        recommendation = {
            "symbol": best_stock["symbol"],
            "rsi": clean_value(best_stock["rsi"], "rsi"),
            "macd": clean_value(best_stock["macd"], "macd"),
            "sma_50": clean_value(best_stock["sma_50"], "sma_50"),
            "sma_200": clean_value(best_stock["sma_200"], "sma_200"),
            "history": {
                "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "prices": [150, 152, 149]
            },
            "news": news
        }

        print(f"📊 Empfehlung gesendet: {recommendation}")
        return recommendation

    except Exception as e:
        print(f"🔥 FEHLER in /recommendation: {str(e)}")
        return {"error": "Internal Server Error", "details": str(e)}

# Startseite auf `index.html` umleiten
@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/api/stock-data")
def get_stock_data(symbol: str = "GOOGL", timeframe: str = "1D"):
    """
    API-Route, die OHLC-Daten (Candlestick) für den ausgewählten Zeitraum liefert.
    Beispiel: /api/stock-data?symbol=GOOGL&timeframe=1D
    """

    # Definiere das Intervall für verschiedene Zeiträume
    timeframe_map = {
        "1D": "1h",
        "1W": "1d",
        "1M": "1d",
        "6M": "1d",
        "1Y": "1wk"
    }

    interval = timeframe_map.get(timeframe, "1d")

    # Holen der Daten von Yahoo Finance
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1y", interval=interval)

    # Falls keine Daten vorhanden sind, Fehler zurückgeben
    if hist.empty:
        return {"error": "Keine Daten gefunden"}

    # Daten formatieren
    data = []
    for index, row in hist.iterrows():
        data.append({
            "date": index.strftime("%Y-%m-%d %H:%M:%S"),
            "open": round(row["Open"], 2),
            "high": round(row["High"], 2),
            "low": round(row["Low"], 2),
            "close": round(row["Close"], 2),
        })

    return data
