import yfinance as yf
import pandas as pd

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")  # 5 Jahre historische Daten
    hist["Ticker"] = ticker
    return hist

def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "market_cap": info.get("marketCap", None),
        "pe_ratio": info.get("trailingPE", None),
        "revenue": info.get("totalRevenue", None),
        "profit_margin": info.get("profitMargins", None),
        "dividend_yield": info.get("dividendYield", None),
    }
