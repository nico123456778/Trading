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
        if sma_50_value is not None and sma_200_value is not None and sma_50_value > sma_200_value:  # Bullisches Signal
            score += 1

        if score > best_score:
            best_stock = indicators
            best_stock["recommendation"] = "BUY"

    return best_stock




# Funktion zur Google News-Abfrage
import urllib.parse

def get_stock_news(symbol):
    try:
        stock = yf.Ticker(symbol)
        company_name = stock.info.get("shortName", symbol)  # Falls kein Name gefunden wird, bleibt Symbol

        query = f"{company_name} Stock Market News OR {company_name} Aktien Nachrichten OR {company_name} Börsennews"
        encoded_query = urllib.parse.quote(query)  # URL-Kodierung hinzufügen
        url = f"https://www.googleapis.com/customsearch/v1?q={encoded_query}&cx=31902bd89c35c40f8&key=AIzaSyBOfkVh3X1lU4LvNExRmVZnEEX2PKuR7KA"

        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            articles = data.get("items", [])  # Holt alle Ergebnisse
            if articles:
                top_news = [f"{article['title']} - {article['link']}" for article in articles[:3]]
                return "; ".join(top_news)  # Gibt bis zu 3 News zurück
    
    except Exception as e:
        print(f"Fehler bei der News-Abfrage für {symbol}: {e}")

    return "Keine aktuellen Nachrichten gefunden."




# Funktion zur Datenbereinigung
def clean_data(value):
    """Überprüft, ob der Wert gültig ist, sonst ersetzt er ihn mit None."""
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value) or value > 1e10 or value < -1e10:
            return None  # Ungültige Werte entfernen
    return value

# API-Endpunkt für Aktienempfehlung
@app.get("/recommendation")
def get_recommendation():
    db = SessionLocal()
    last_recommendation = db.query(StockRecommendation).order_by(StockRecommendation.date.desc()).first()

    # Falls die letzte Empfehlung älter als 12 Stunden ist, neue Berechnung starten
    if not last_recommendation or (datetime.utcnow() - last_recommendation.date) > timedelta(hours=12):
        best_stock = select_best_stock()

        if best_stock:
            news = get_stock_news(best_stock["symbol"])
            new_recommendation = StockRecommendation(
                id=str(datetime.utcnow().timestamp()),
                symbol=best_stock["symbol"],
                recommendation=best_stock["recommendation"],
                rsi=clean_data(best_stock["rsi"]),
                macd=clean_data(best_stock["macd"]),
                sma_50=clean_data(best_stock["sma_50"]),
                sma_200=clean_data(best_stock["sma_200"]),
                news=news,
                date=datetime.utcnow()
            )
            db.add(new_recommendation)
            db.commit()
            db.refresh(new_recommendation)
            db.close()
            return new_recommendation

    db.close()
    return last_recommendation

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Statische Dateien (z. B. index.html) bereitstellen
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

