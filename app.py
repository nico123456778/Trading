import os
import time
import pandas as pd
import yfinance as yf
import requests
from fastapi import FastAPI
from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import math

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
            continue

        # Falls Werte Pandas-Serien sind, extrahiere den letzten Wert mit iloc[-1]
        rsi_value = indicators["rsi"]
        macd_value = indicators["macd"]
        sma_50_value = indicators["sma_50"]
        sma_200_value = indicators["sma_200"]

        if isinstance(rsi_value, pd.Series):
            rsi_value = float(rsi_value.iloc[-1])

        if isinstance(macd_value, pd.Series):
            macd_value = float(macd_value.iloc[-1])

        if isinstance(sma_50_value, pd.Series):
            sma_50_value = float(sma_50_value.iloc[-1])

        if isinstance(sma_200_value, pd.Series):
            sma_200_value = float(sma_200_value.iloc[-1])

        # Scoring-System für die Aktienauswahl
        score = 0
        if rsi_value is not None and rsi_value < 30:  # Überverkauftes Signal
            score += 2
        if macd_value is not None and macd_value > 0:  # Positiver MACD-Trend
            score += 1
