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
from textblob import TextBlob

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

# Aktienliste mit S&P 500 und internationalen Aktien
STOCK_LIST = sp500_tickers + dax40_tickers + intl_tickers

# AI-Modell laden
model = joblib.load("models/stock_model.pkl")

# Funktion zur Sentiment-Analyse von Finanznachrichten
def get_news_sentiment(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey=DEIN_API_KEY"
    response = requests.get(url).json()
    news = response.get("articles", [])[:5]
    
    sentiment_scores = []
    for article in news:
        text = article.get("title", "") + " " + article.get("description", "")
        sentiment = TextBlob(text).sentiment.polarity
        sentiment_scores.append(sentiment)
    
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

# Funktion zur Berechnung technischer Indikatoren und Fundamentaldaten
def calculate_indicators(symbol):
    print(f"🔍 Berechne Indikatoren für: {symbol}")
    df = yf.download(symbol, period="1y", interval="1d")
    stock = yf.Ticker(symbol)
    info = stock.info
    
    if df.empty:
        print(f"⚠ Keine Daten für {symbol} gefunden!")
        return None

    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()
    df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(14).mean() / df["Close"].pct_change().rolling(14).std()))
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["SMA200"].fillna(method="ffill", inplace=True)
    latest_data = df.iloc[-1]
    
    result = {
        "symbol": symbol,
        "RSI": latest_data["RSI"],
        "MACD": latest_data["MACD"],
        "SMA50": latest_data["SMA50"],
        "SMA200": latest_data["SMA200"],
        "Revenue": info.get("totalRevenue", None),
        "NetIncome": info.get("netIncome", None),
        "PE_Ratio": info.get("trailingPE", None),
        "DividendYield": info.get("dividendYield", None),
        "Sentiment": get_news_sentiment(symbol)
    }
    print(f"✅ Indikatoren für {symbol}: {result}")
    return result

# Empfehlungssystem mit Sentiment und Fundamentaldaten@app.get("/recommendation")
def get_best_stock():
    try:
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
            if indicators["Sentiment"] > 0:
                score += 2
            if ai_signal == 1:
                score += 3
            if score > best_score:
                best_score = score
                best_stock = indicators
        return best_stock if best_stock else {"error": "Keine Empfehlung verfügbar"}
    except Exception as e:
        print(f"🔥 FEHLER in /recommendation: {str(e)}")
        return {"error": "Internal Server Error", "details": str(e)}

@app.get("/")
def read_root():
    return FileResponse("static/index.html")
