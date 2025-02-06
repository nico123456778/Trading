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

# Funktion zur Berechnung technischer Indikatoren
def calculate_indicators(symbol):
    print(f"🔍 Berechne Indikatoren für: {symbol}")
    df = yf.download(symbol, period="6mo", interval="1d")
    if df.empty:
        print(f"⚠ Keine Daten für {symbol} gefunden!")
        return None

    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["SMA_200"].fillna(df["SMA_50"], inplace=True)
    df["SMA_200"].fillna(df["Close"].rolling(200).mean(), inplace=True)
    df["SMA_200"].fillna(df["Close"].mean(), inplace=True)
    
    df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(14).mean() / df["Close"].pct_change().rolling(14).std()))
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    latest_data = df.iloc[-1]
    result = {
        "rsi": safe_float(latest_data["RSI"], "RSI"),
        "macd": safe_float(latest_data["MACD"], "MACD"),
        "sma_50": safe_float(latest_data["SMA_50"], "SMA_50"),
        "sma_200": safe_float(latest_data["SMA_200"], "SMA_200"),
    }
    print(f"✅ Indikatoren für {symbol}: {result}")
    return result

# AI-Modell-Vorhersage
def ai_predict(indicators):
    df = pd.DataFrame([indicators])
    prediction = model.predict(df)
    return int(prediction[0])

@app.get("/recommendation")
def get_best_stock():
    try:
        best_stock = select_best_stock()
        if not best_stock:
            return {"message": "❌ Keine passende Aktie gefunden."}
        recommendation = clean_json_data({
            "rsi": best_stock["rsi"],
            "macd": best_stock["macd"],
            "sma_50": best_stock["sma_50"],
            "sma_200": best_stock["sma_200"],
            "ai_signal": ai_predict(best_stock)
        })
        return recommendation
    except Exception as e:
        print(f"🔥 FEHLER in /recommendation: {str(e)}")
        return {"error": "Internal Server Error", "details": str(e)}
