
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

   import math

def clean_data(value):
    """Überprüft, ob der Wert gültig ist, sonst ersetzt er ihn mit None."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value

    @app.get("/recommendation")
def get_recommendation():
    best_stock = select_best_stock()

    if best_stock is None:
        return {"message": "Keine gültige Empfehlung gefunden"}

    return {
        "symbol": best_stock["symbol"],
        "rsi": clean_data(best_stock.get("rsi")),
        "macd": clean_data(best_stock.get("macd")),
        "sma_50": clean_data(best_stock.get("sma_50")),
        "sma_200": clean_data(best_stock.get("sma_200")),
        "recommendation": best_stock.get("recommendation", "Keine Empfehlung")
    }

    

