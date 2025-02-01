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

# Datenbank einrichten (SQLite für einfaches Deployment)
DATABASE_URL = "sqlite:///./stocks.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Datenbankmodell für Aktienempfehlung
class StockRecommendation(Base):
    __tablename__ = "stock_recommendations"

    id = Column(String, primary_key=True, index=True)
    symbol = Column(String, index=True)
    recommendation = Column(String)
    rsi = Column(Float)
    macd = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    news = Column(String)
    date = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Aktienliste (S&P 500 + dynamische Auswahl)
STOCK_LIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "AMD", "BA"]

# Google Custom Search API für News
GOOGLE_API_KEY = "AIzaSyBOfkVh3X1lU4LvNExRmVZnEEX2PKuR7KA"
GOOGLE_CSE_ID = "inlaid-water-449413-s6"

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
            print(f"❌ Keine Daten für {stock}")
            continue

        # Falls Werte Pandas-Serien sind, extrahiere den letzten Wert mit iloc[-1]
        for key in ["rsi", "macd", "sma_50", "sma_200"]:
            if isinstance(indicators[key], pd.Series):
                indicators[key] = indicators[key].iloc[-1]

        # Debugging: Zeigt die Indikatoren und den Score
        print(f"📊 Daten für {stock}: {indicators}")

        # Beispiel-Scoring
        score = 0
        if indicators["rsi"] is not None and indicators["rsi"] < 30:  
            score += 2
        if indicators["macd"] is not None and indicators["macd"] > 0:  
            score += 1

        # Wähle die beste Aktie
        if score > best_score:
            best_score = score
            best_stock = indicators

    if best_stock:
        print(f"✅ Beste Aktie: {best_stock['symbol']} mit Score {best_score}")
    else:
        print("❌ Keine Aktie gefunden")

    return best_stock

# API-Endpunkt für die beste Aktie
@app.get("/recommendation")
def get_best_stock():
    best_stock = select_best_stock()  # Wählt die beste Aktie aus
    if not best_stock:
        return {"error": "Keine Empfehlung verfügbar"}

    # Nachrichten auswerten (später mit KI)
    news = [
        {"title": "Marktanalyse: Warum AAPL stark ansteigt", "rating": 9},
        {"title": "Analysten sehen Potenzial für MSFT", "rating": 8},
    ]

    return {
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

# API-Endpunkt für Aktienliste
@app.get("/stocks")
def get_stock_data():
    stock_data = []

    for stock in STOCK_LIST:
        print(f"📡 Fetching data for {stock}...")  # Debugging
        indicators = calculate_indicators(stock)

        if indicators:
            print(f"✅ Data received for {stock}: {indicators}")  # Debugging

            # Erstelle sicheres Dictionary ohne Ticker-Objekte oder NaN/Inf-Werte
            clean_indicators = {}

            for key, value in indicators.items():
                if hasattr(value, "iloc"):  
                    value = value.iloc[-1]  # Letzten Wert aus Pandas-Serie nehmen

                if isinstance(value, (float, np.float32, np.float64)):
                    if np.isnan(value) or np.isinf(value):
                        print(f"⚠ WARNUNG: {stock} hat ungültigen Wert bei {key}: {value}")
                        clean_indicators[key] = None  # Ersetze ungültige Werte
                    else:
                        clean_indicators[key] = round(value, 6)  # Runden für JSON-Sicherheit
                else:
                    clean_indicators[key] = value  

            stock_data.append(clean_indicators)  

    print(f"📊 FINAL RETURN DATA: {stock_data}")  
    return {"stocks": stock_data}

# Statische Dateien bereitstellen
app.mount("/static", StaticFiles(directory="static"), name="static")

# Startseite auf `index.html` umleiten
@app.get("/")
def read_root():
    return FileResponse("static/index.html")
