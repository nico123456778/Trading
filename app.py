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
from textblob import TextBlob

# FastAPI App erstellen
app = FastAPI()

# Datenbank einrichten
DATABASE_URL = "sqlite:///./stocks.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Aktienliste
STOCK_LIST = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX", "AMD", "BA"]

# Indikatoren berechnen
def calculate_indicators(symbol):
    df = yf.download(symbol, period="6mo", interval="1d")
    if df.empty:
        return None
    
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + df["Close"].pct_change().rolling(14).mean()))
    
    return {
        "SMA50": round(df["SMA50"].iloc[-1], 2),
        "SMA200": round(df["SMA200"].iloc[-1], 2),
        "MACD": round(df["MACD"].iloc[-1], 2),
        "RSI": round(df["RSI"].iloc[-1], 2),
    }

# News-Sentiment analysieren
def analyze_news_sentiment(stock):
    API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
    SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    query = f"{stock} stock news"
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={SEARCH_ENGINE_ID}"
    
    response = requests.get(url).json()
    sentiment_score = 0
    
    if "items" in response:
        for item in response["items"][:5]:
            text = item.get("snippet", "")
            sentiment_score += TextBlob(text).sentiment.polarity
    
    return round(sentiment_score / 5, 2) if sentiment_score else 0

# Beste Aktie auswählen
def select_best_stock():
    best_stock = None
    best_score = -999
    best_indicators = None
    best_sentiment = 0
    
    for stock in STOCK_LIST:
        indicators = calculate_indicators(stock)
        sentiment = analyze_news_sentiment(stock)
        
        if not indicators:
            continue
        
        score = 0
        if indicators["RSI"] < 30: score += 10
        if indicators["MACD"] > 0: score += 10
        if indicators["SMA50"] > indicators["SMA200"]: score += 15
        if sentiment > 0.1: score += 10  # Positiver News-Sentiment
        
        if score > best_score:
            best_stock = stock
            best_score = score
            best_indicators = indicators
            best_sentiment = sentiment
    
    return best_stock, best_indicators, best_sentiment

@app.get("/recommendation")
def get_recommendation():
    stock, indicators, sentiment = select_best_stock()
    return {
        "recommended_stock": stock,
        "rsi": indicators["RSI"],
        "macd": indicators["MACD"],
        "sma50": indicators["SMA50"],
        "sma200": indicators["SMA200"],
        "sentiment": sentiment,
    }
