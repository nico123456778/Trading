import os
import time
import pandas as pd
import yfinance as yf
import requests
import numpy as np
from fastapi import FastAPI
from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import math
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import joblib

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

# AI-Modell laden
model = joblib.load("models/stock_model.pkl")

# Sicheres Konvertieren von Werten
def safe_float(value, key):
    if isinstance(value, pd.Series):
        print(f"⚠ WARNUNG: {key} ist eine Pandas-Serie! Extrahiere letzten Wert.")
        value = value.iloc[-1]
    if isinstance(value, (float, int)):
        return float(value)
    print(f"❌ FEHLER: {key} konnte nicht in Float umgewandelt werden! Setze auf None.")
    return None

# Funktion zur Bereinigung ungültiger Werte
def clean_value(value, key):
    if isinstance(value, (float, np.float32, np.float64)):
        if np.isnan(value) or np.isinf(value):
            print(f"⚠ WARNUNG: Ungültiger Wert bei {key}: {value}, wird auf None gesetzt")
            return None
        return round(value, 6)
    return value

# Rekursive Bereinigung von JSON-Daten
def clean_json_data(data):
    if isinstance(data, dict):
        return {key: clean_json_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_json_data(item) for item in data]
    elif isinstance(data, (float, np.float32, np.float64)):
        if np.isnan(data) or np.isinf(data):
            print(f"⚠ WARNUNG: Ungültiger Wert erkannt: {data}")
            return None
        return round(data, 6)
    return data

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
    print(f"🔍 Berechne Indikatoren für: {symbol}")
    df = yf.download(symbol, period="6mo", interval="1d")
    if df.empty:
        print(f"⚠ Keine Daten für {symbol} gefunden!")
        return None

    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()
    
    # Debugging-Log für SMA200 prüfen
    print(f"📊 {symbol}: Datenpunkte verfügbar = {len(df)} | Letzter SMA200-Wert: {df['SMA200'].iloc[-1]}")
    
    # Falls SMA200 NaN ist, nutze SMA50 oder den Durchschnitt der letzten 200 Tage
    df["SMA200"].fillna(df["SMA50"], inplace=True)  # Falls möglich, SMA50 nutzen
    df["SMA200"].fillna(df["Close"].rolling(200).mean(), inplace=True)  # Falls noch NaN, Durchschnitt der letzten 200 Tage
    df["SMA200"].fillna(df["Close"].mean(), inplace=True)  # Falls alles fehlschlägt, nimm den Gesamt-Durchschnitt

    df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(14).mean() / df["Close"].pct_change().rolling(14).std()))
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    latest_data = df.iloc[-1]
    result = {
        "RSI": safe_float(latest_data["RSI"], "RSI"),
        "MACD": safe_float(latest_data["MACD"], "MACD"),
        "SMA50": safe_float(latest_data["SMA50"], "SMA50"),
        "SMA200": safe_float(latest_data["SMA200"], "SMA200"),
    }
    print(f"✅ Indikatoren für {symbol}: {result}")
    return result

@app.get("/recommendation")
def get_best_stock():
    best_stock = select_best_stock()
    if not best_stock:
        return {"error": "Keine Empfehlung verfügbar"}
    return clean_json_data(best_stock)

@app.get("/")
def read_root():
    return FileResponse("static/index.html")
