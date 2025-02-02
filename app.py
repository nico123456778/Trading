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

# API-Endpunkt für die beste Aktie
@app.get("/recommendation")
def get_best_stock():
    best_stock = select_best_stock()
    if not best_stock:
        return {"error": "Keine Empfehlung verfügbar"}
    
    news_response = search(best_stock["symbol"] + " stock news")
    news = news_response.get("items", []) if news_response else []

    return {
        "symbol": best_stock["symbol"],
        "rsi": best_stock["rsi"],
        "macd": best_stock["macd"],
        "sma_50": best_stock["sma_50"],
        "sma_200": best_stock["sma_200"],
        "google_api_key": GOOGLE_API_KEY,
        "google_cse_id": GOOGLE_CSE_ID,
        "news": news
    }

# Startseite auf `index.html` umleiten
@app.get("/")
def read_root():
    return FileResponse("static/index.html")
