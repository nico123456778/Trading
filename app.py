
import os
import time
import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

app = FastAPI()

DATABASE_URL = "sqlite:///./stocks.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class StockRecommendation(Base):
    __tablename__ = "stock_recommendations"

    id = Column(String, primary_key=True, index=True)
    symbol = Column(String, index=True)
    recommendation = Column(String)
    rsi = Column(Float)
    macd = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    date = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

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

@app.get("/recommendation")
def get_recommendation():
    best_stock = calculate_indicators("AAPL")
    return best_stock
