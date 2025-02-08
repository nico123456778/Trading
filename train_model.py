import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import yfinance as yf
import ta
import os
import requests
from textblob import TextBlob

# 📥 Automatische Aktien- und Krypto-Listen
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
table = pd.read_html(requests.get(url).text)[0]
sp500_tickers = table["Symbol"].tolist()

dax40_tickers = ["ADS.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "CON.DE", "DAI.DE", "DBK.DE", "DPW.DE", "DTE.DE", "EOAN.DE", "FME.DE", "FRE.DE", "HEI.DE", "HEN3.DE", "IFX.DE", "LHA.DE", "LIN.DE", "MRK.DE", "MUV2.DE", "RWE.DE", "SAP.DE", "SIE.DE", "TKA.DE", "VNA.DE", "VOW3.DE"]

crypto_tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "DOT-USD", "MATIC-USD", "XLM-USD"]

tickers = sp500_tickers + dax40_tickers + crypto_tickers

# 📊 Makroökonomische Faktoren abrufen
def get_interest_rate():
    try:
        fed_funds = yf.Ticker("^IRX")
        return fed_funds.history(period="1mo")["Close"].iloc[-1]
    except:
        return None

def get_inflation():
    try:
        cpi = yf.Ticker("^CPI")
        return cpi.history(period="1mo")["Close"].iloc[-1]
    except:
        return None

inflation = get_inflation()
interest_rate = get_interest_rate()

all_data = []
sp500 = yf.download("^GSPC", period="5y", interval="1d")["Close"].squeeze()

for ticker in tickers:
    print(f"Lade Daten für {ticker}...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")
    
    if hist.empty:
        continue
    
    hist.index = hist.index.tz_localize(None)  # Zeitzonen entfernen
    info = stock.info
    sentiment = TextBlob(info.get("longBusinessSummary", "")).sentiment.polarity if "longBusinessSummary" in info else 0
    
    hist["Ticker"] = ticker
    hist["RSI"] = ta.momentum.RSIIndicator(hist["Close"]).rsi()
    hist["MACD"] = ta.trend.MACD(hist["Close"]).macd()
    hist["SMA50"] = ta.trend.SMAIndicator(hist["Close"], window=50).sma_indicator()
    hist["SMA200"] = ta.trend.SMAIndicator(hist["Close"], window=200).sma_indicator()
    hist["ATR"] = ta.volatility.AverageTrueRange(hist["High"], hist["Low"], hist["Close"]).average_true_range()
    hist["BB_Upper"] = ta.volatility.BollingerBands(hist["Close"]).bollinger_hband()
    hist["BB_Lower"] = ta.volatility.BollingerBands(hist["Close"]).bollinger_lband()
    hist["Volume_Trend"] = hist["Volume"] / hist["Volume"].rolling(10).mean()
    hist["Relative_Strength"] = hist["Close"] / sp500.ffill().reindex(hist.index, method="nearest")
    hist["Sentiment"] = sentiment
    hist["Inflation"] = inflation
    hist["Interest_Rate"] = interest_rate
    
    hist = hist.dropna(thresh=8)  # Fehlende Werte minimieren
    all_data.append(hist)

if len(all_data) == 0:
    raise ValueError("🚨 Keine Daten geladen! Überprüfe die Tickerauswahl oder Internetverbindung.")

df = pd.concat(all_data)
df.to_csv("stock_data.csv", index=False)

# 📊 Optimierte Kaufbedingungen
X = df.drop(columns=["Ticker"])
y = (df["RSI"] < 50) & (df["MACD"] > -3) & (df["Sentiment"] > -0.8) & (df["Relative_Strength"] > 0.7) & (df["SMA50"] > df["SMA200"]) & (df["Volume_Trend"] > 0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "stock_model.pkl")
print("✅ Modell wurde erfolgreich trainiert und gespeichert!")
