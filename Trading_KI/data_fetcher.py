import yfinance as yf
import pandas as pd
from fastapi import FastAPI, Query

app = FastAPI()

# Funktion zum Abrufen historischer Kursdaten
def get_stock_data(ticker: str, period: str = "1mo"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval="1d")
    if hist.empty:
        return {"error": "Keine Daten gefunden"}
    return {
        "dates": hist.index.strftime("%Y-%m-%d").tolist(),
        "prices": hist["Close"].tolist()
    }

@app.get("/chart")
def get_chart(symbol: str = Query(...), timeframe: str = Query("1mo")):
    return get_stock_data(symbol, timeframe)
