import os
import pandas as pd
import pickle
import yfinance as yf
import requests
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from textblob import TextBlob
import ta  # Technische Analyse Bibliothek
import time
import threading
import time

global_scores = []

CACHE_FILE = "cached_stock_data.csv"


def load_cached_data():
    """Lädt gespeicherte Daten, falls vorhanden"""
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE)
    return pd.DataFrame(columns=["Ticker", "Close"])


def fetch_data_with_cache(ticker):
    """Lädt Aktien-Daten nur, wenn sie nicht bereits gespeichert wurden"""
    existing_data = load_cached_data()

    if ticker in existing_data["Ticker"].values:
        print(f"⏩ {ticker} bereits gespeichert, überspringe Abruf.")
        return

    try:
        print(f"📊 Lade Daten für {ticker} ...")
        data = yf.download(ticker, period="7d", interval="1d")

        if data.empty or "Close" not in data.columns:
            print(f"⚠️ Keine Daten für {ticker}, überspringe...")
            return

        # Speichere nur den letzten Schlusskurs
        new_data = pd.DataFrame({
            "Ticker": [ticker],
            "Close": [data["Close"].iloc[-1]]
        })

        new_data.to_csv(CACHE_FILE, mode="a", header=not os.path.exists(CACHE_FILE), index=False)
        print(f"✅ {ticker} gespeichert.")

    except Exception as e:
        print(f"❌ Fehler bei {ticker}: {e}")

    time.sleep(2)  # 🔥 Wartezeit, um API-Sperren zu vermeiden


# FastAPI App erstellen
app = FastAPI()

# CORS aktivieren, falls API von einer externen Website aufgerufen wird
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google API Umgebungsvariablen aus Render laden
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


# Liste der zu analysierenden Aktien und Kryptowährungen
stock_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK.B", "JPM",
              "V", "JNJ", "PG", "UNH", "XOM", "HD", "MA", "PFE", "ABBV", "CVX", "KO",
              "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "DOT-USD"]

print("Folgende Werte werden von der KI analysiert:")
for ticker in stock_list:
    print("-", ticker)
print(f"Gesamtanzahl: {len(stock_list)}")


# KI-Modell laden mit Debugging
model_path = "Trading_KI/models/stock_model.pkl"

if not os.path.exists(model_path):
    print("🚨 Fehler: Die Datei 'stock_model.pkl' existiert nicht im Verzeichnis!")
    model = None
else:
    try:
        model = joblib.load(model_path)
        print("✅ Modell erfolgreich geladen!")
    except Exception as e:
        print(f"❌ Fehler beim Laden des Modells: {e}")
        model = None


# Funktion zur Sentiment-Analyse von Finanznachrichten
def get_news_sentiment(query):
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={GOOGLE_CSE_ID}&key={GOOGLE_API_KEY}"
        response = requests.get(url).json()
        news = response.get("items", [])[:5]

        sentiment_scores = []
        for article in news:
            snippet = article.get("snippet", "")
            analysis = TextBlob(snippet)
            sentiment_scores.append(analysis.sentiment.polarity)

        if sentiment_scores:
            return sum(sentiment_scores) / len(sentiment_scores)
    except Exception:
        pass
    return 0.0  # Fallback-Sentiment


# Auswahl der besten Aktie/Krypto
def select_best_asset():
    global global_scores  # WICHTIG: Globale Variable setzen
    scores = []  # Lokale Liste für Berechnungen

    for ticker in stock_list:
        try:
            print(f"📊 Lade Daten für {ticker} ...")  # Debugging-Log
            data = yf.download(ticker, period="7d", interval="1d")

            if data is None or data.empty or "Close" not in data.columns:
                print(f"⚠️ Keine gültigen Daten für {ticker} erhalten, überspringe...")
                continue

            print(data.tail())  # Debugging-Log für letzte Zeilen der Daten

            # Berechnung der technischen Indikatoren
            df = pd.DataFrame({
                "Close": float(data["Close"].iloc[-1]),
                "RSI": float(ta.momentum.RSIIndicator(data["Close"]).rsi().dropna().values[-1]),
                "MACD": float(ta.trend.MACD(data["Close"]).macd().dropna().values[-1]),
                "SMA50": float(ta.trend.SMAIndicator(data["Close"], window=50).sma_indicator().dropna().values[-1]),
                "SMA200": float(ta.trend.SMAIndicator(data["Close"], window=200).sma_indicator().dropna().values[-1]),
            }, index=[0])

            print(f"📊 Berechnete Indikatoren für {ticker}: {df.to_dict(orient='records')}")  # Debugging
            prediction = 0

            if model and not df.isnull().values.any():
                prediction = model.predict(df)[0]
                sentiment = get_news_sentiment(ticker)
                final_score = prediction + sentiment

                print(f"🤖 KI-Einschätzung für {ticker}: Prediction={prediction}, Sentiment={sentiment}, Final Score={final_score}")  # Debugging

                scores.append((ticker, final_score))
                global_scores.append((ticker, final_score))

        except Exception as e:
            print(f"❌ Fehler bei der Modellvorhersage: {e}")

    if scores:
        best_asset = max(scores, key=lambda x: x[1])
        print(f"🏆 Beste Aktie/Krypto: {best_asset[0]} mit Score {best_asset[1]}")  # Debugging
        return best_asset
    else:
        print("⚠️ Keine geeignete Aktie/Krypto gefunden.")
        return None, 0.0



def fetch_all_data():
    print("🚀 Starte das Laden der Aktien...")
    for ticker in stock_list:
        fetch_data_with_cache(ticker)
        wait_time = random.uniform(1, 3)  # 🔥 Verhindert API-Sperren
        print(f"⏳ Warte {wait_time:.2f} Sekunden, um API-Sperren zu vermeiden...")
        time.sleep(wait_time)
    print("✅ Alle Aktien-Daten wurden geladen!")

# Hintergrund-Thread starten
thread = threading.Thread(target=fetch_all_data, daemon=True)
thread.start()

# 🔥 Endlosschleife, damit die App nicht stoppt
while True:
    time.sleep(10)  # App bleibt aktiv


@app.get("/")
def get_recommended_stock():
    """ Gibt die beste Aktie aus der Analyse zurück """
    global global_scores  # WICHTIG: Globale Variable verwenden!

    if not global_scores:
        return {"error": "Keine Daten verfügbar"}

    best_stock = max(global_scores, key=lambda x: x[1], default=(None, 0))

    return {
        "ticker": best_stock[0],
        "indikatoren": {
            "RSI": best_stock[1],
        }
    }



