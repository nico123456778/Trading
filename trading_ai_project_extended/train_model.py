import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import yfinance as yf
import ta

# 📥 Aktienliste (z. B. S&P 100, um Performance zu optimieren)
tickers = ["AAPL", "GOOGL", "AMZN", "TSLA", "MSFT", "NVDA", "META", "NFLX", "JPM", "BA"]

all_data = []

for ticker in tickers:
    print(f"Lade Daten für {ticker}...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")

    if hist.empty:
        continue

    hist["Ticker"] = ticker
    hist["RSI"] = ta.momentum.RSIIndicator(hist["Close"]).rsi()
    hist["MACD"] = ta.trend.MACD(hist["Close"]).macd()
    hist["SMA50"] = hist["Close"].rolling(window=50).mean()
    hist["SMA200"] = hist["Close"].rolling(window=200).mean()
    hist["Buy_Signal"] = (hist["RSI"] < 30) & (hist["MACD"] > 0) & (hist["SMA50"] > hist["SMA200"])
    
    hist = hist.dropna()
    all_data.append(hist)

df = pd.concat(all_data)
df.to_csv(os.path.join(project_dir, "data", "stock_data.csv"), index=False)

# 📊 Modelltraining
X = df[["RSI", "MACD", "SMA50", "SMA200"]]
y = df["Buy_Signal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 💾 Modell speichern
joblib.dump(model, os.path.join(project_dir, "models", "stock_model.pkl"))
print("✅ Modell wurde trainiert und gespeichert!")
