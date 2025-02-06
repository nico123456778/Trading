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
import joblib

# FastAPI App erstellen
app = FastAPI()

# Statische Dateien bereitstellen
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

# Google API Umgebungsvariablen
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Datenbank einrichten
DATABASE_URL = "sqlite:///./stocks.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Aktienliste
STOCK_LIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "AMD", "BA"]

# AI-Modell laden
model = joblib.load("models/stock_model.pkl")

# Funktion zum sicheren Konvertieren von Werten
def safe_float(value, key):
    if isinstance(value, pd.Series):
        print(f"⚠ WARNUNG: {key} ist eine Pandas-Serie! Extrahiere letzten Wert.")
        value = value.iloc[-1]
    if isinstance(value, (float, int)):
        return float(value)
    print(f"❌ FEHLER: {key} konnte nicht in Float umgewandelt werden! Setze auf None.")
    return None

# Funktion zur Berechnung technischer Indikatoren
def calculate_indicators(symbol):
    print(f"🔍 Berechne Indikatoren für: {symbol}")
    df = yf.download(symbol, period="6mo", interval="1d")
    if df.empty:
        print(f"⚠ Keine Daten für {symbol} gefunden!")
        return None

    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(14).mean() / df["Close"].pct_change().rolling(14).std()))
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    latest_data = df.iloc[-1]
    return {
        "symbol": symbol,
        "MACD": safe_float(latest_data["MACD"], "MACD"),
        "RSI": safe_float(latest_data["RSI"], "RSI"),
        "SMA200": safe_float(latest_data["SMA_200"], "SMA_200"),
        "SMA50": safe_float(latest_data["SMA_50"], "SMA_50")
    }

# AI-Modell-Vorhersage
def ai_predict(symbol, indicators):
    df = pd.DataFrame([indicators])
    prediction = model.predict(df)
    return int(prediction[0])

# Funktion zur Auswahl der besten Aktie
def select_best_stock():
    best_stock = None
    best_score = float('-inf')
    for stock in STOCK_LIST:
        indicators = calculate_indicators(stock)
        if not indicators:
            continue
        ai_signal = ai_predict(stock, indicators)
        score = 0
        if indicators["RSI"] is not None and indicators["RSI"] < 30:
            score += 2
        if indicators["MACD"] is not None and indicators["MACD"] > 0:
            score += 1
        if ai_signal == 1:
            score += 3
        if score > best_score:
            best_score = score
            best_stock = indicators
    return best_stock

@app.get("/recommendation")
def get_best_stock():
    try:
        best_stock = select_best_stock()
        if not best_stock:
            return {"error": "Keine Empfehlung verfügbar"}

        print(f"DEBUG: best_stock = {best_stock}")

        recommendation = {
            "symbol": best_stock["symbol"],
            "MACD": best_stock["MACD"],
            "RSI": best_stock["RSI"],
            "SMA200": best_stock["SMA200"],
            "SMA50": best_stock["SMA50"],
            "ai_signal": ai_predict(best_stock["symbol"], best_stock)
        }

        print(f"DEBUG: Features an Modell: {recommendation}")
        return recommendation

    except Exception as e:
        print(f"🔥 FEHLER in /recommendation: {str(e)}")
        return {"error": "Internal Server Error", "details": str(e)}

@app.get("/")
def read_root():
    return {"message": "Trading App läuft!"}
