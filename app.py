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
import random


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
stock_list = [
    # S&P 500 Aktien
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "JPM",
    "V", "JNJ", "PG", "UNH", "XOM", "HD", "MA", "PFE", "ABBV", "CVX", "KO",
    "PEP", "MRK", "BAC", "DIS", "NFLX", "INTC", "CMCSA", "ADBE", "T", "CRM",
    "MCD", "COST", "NKE", "WMT", "IBM", "QCOM", "HON", "ORCL", "AMD", "CAT",
    "BA", "MMM", "GE", "TXN", "LMT", "GS", "USB", "PYPL", "F", "GM", "DUK",
    "SO", "SBUX", "CSCO", "MDT", "NEE", "UPS", "LOW", "TGT", "AXP", "CI",
    "BLK", "REGN", "VRTX", "ADP", "DE", "ISRG", "CVS", "ETN", "SPGI", "ICE",
    "NOW", "GILD", "CME", "MO", "HUM", "PNC", "DHR", "SCHW", "AMT", "TMO",
    
    # DAX 40 Aktien
    "SAP.DE", "DTE.DE", "SIE.DE", "BAYN.DE", "VOW3.DE", "ALV.DE", "BAS.DE",
    "ADS.DE", "BMW.DE", "FRE.DE", "MUV2.DE", "RWE.DE", "LIN.DE", "DBK.DE",
    "DHER.DE", "BEI.DE", "MTX.DE", "HEN3.DE", "HFG.DE", "HEI.DE", "CON.DE",
    "ENR.DE", "1COV.DE", "SHL.DE", "ZAL.DE", "SY1.DE", "AIR.DE", "IFX.DE",
    "PUM.DE", "WDI.DE", "LEG.DE", "EVD.DE", "MRK.DE", "TLX.DE", "BNR.DE",
    "RIB.DE", "WCH.DE", "TEG.DE", "EVK.DE", "HNR1.DE",
    
    # Internationale Aktien
    "BABA", "TSM", "NIO", "TCEHY", "SNY", "ASML", "RY", "TD", "SHOP", "SU",
    "BIDU", "SNP", "BP", "RIO", "BHP", "NVS", "ORAN", "SIEGY", "EONGY",
    "UBS", "CSGN", "ABB", "NESN", "RDSB", "BP.L", "HSBA.L", "BARC.L", "LLOY.L",
    
    # Kryptowährungen
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "DOT-USD", "MATIC-USD",
    "XLM-USD", "ADA-USD", "DOGE-USD", "LTC-USD", "BCH-USD", "UNI-USD", "LINK-USD",
    "XMR-USD", "ETC-USD", "VET-USD", "FIL-USD", "TRX-USD", "EOS-USD", "XTZ-USD",
    "ATOM-USD", "AAVE-USD", "MKR-USD", "ALGO-USD", "DASH-USD", "ZEC-USD"
]


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
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        print("⚠️ Google API Key fehlt! Sentiment-Analyse deaktiviert.")
        return 0.0  # Falls kein API-Key, dann neutraler Wert
    
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
    except Exception as e:
        print(f"❌ Fehler bei Sentiment-Analyse: {e}")

    return 0.0  # Fallback-Sentiment



# Auswahl der besten Aktie/Krypto
def select_best_asset():
    global global_scores  # WICHTIG: Globale Variable setzen
    global_scores.clear()  # 🔥 ALTE WERTE LÖSCHEN, DAMIT NUR DIE NEUSTE ANALYSE DRIN BLEIBT
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


if df["RSI"].iloc[0] < 30:
    print(f"⚠️ {ticker} ist stark überverkauft! Möglicher Kauf-Kandidat.")
elif df["RSI"].iloc[0] > 70:
    print(f"⚠️ {ticker} ist stark überkauft! Risiko beachten.")

if df["MACD"].iloc[0] > 0:
    print(f"✅ {ticker} zeigt ein bullisches Signal laut MACD.")


print(f"📊 Berechnete Indikatoren für {ticker}: {df.to_dict(orient='records')}")  # Debugging

### 🔥 NEU: Prüfen, ob es `NaN`-Werte gibt, die das Modell blockieren könnten
if df.isnull().values.any():
    print(f"⚠️ NaN-Werte gefunden für {ticker}: {df.to_dict(orient='records')}")
    continue  # Aktie wird übersprungen, falls NaN enthalten ist

prediction = 0

if model:
    ### 🔥 NEU: Test-Prediction mit Dummy-Werten
    try:
        test_df = pd.DataFrame({
            "Close": [100],
            "RSI": [50],
            "MACD": [0.5],
            "SMA50": [98],
            "SMA200": [95],
        })
        test_prediction = model.predict(test_df)[0]
        print(f"✅ Test-Prediction des Modells (sollte sinnvoll sein): {test_prediction}")
    except Exception as e:
        print(f"❌ Fehler bei Test-Prediction: {e}")

    ### 🔥 Falls das Modell gültig ist, Vorhersage für echte Daten durchführen
    prediction = model.predict(df)[0]
    sentiment = get_news_sentiment(ticker)
if sentiment is None:
    sentiment = 0.1  # Falls kein Sentiment gefunden, geben wir eine kleine positive Bewertung

final_score = prediction + sentiment  # 🔥 final_score IMMER berechnen!


# Falls RSI unter 50 ist (aber nicht zu niedrig), geben wir einen kleinen Bonus
if df["RSI"].iloc[0] < 50:
    final_score += 0.2  

# Falls MACD positiv ist, geben wir einen noch stärkeren Bonus
if df["MACD"].iloc[0] > 0:
    final_score += 0.3  


    print(f"🤖 KI-Einschätzung für {ticker}: Prediction={prediction}, Sentiment={sentiment}, Final Score={final_score}")  # Debugging

    scores.append((ticker, final_score))
    global_scores.append((ticker, final_score))

        except Exception as e:
            print(f"❌ Fehler bei der Modellvorhersage: {e}")

if scores:
    best_asset = max(scores, key=lambda x: x[1])

    # Falls alle Scores unter 0 sind, trotzdem eine Empfehlung ausgeben
    if best_asset[1] < 0:
        print(f"⚠️ Alle Scores sind niedrig, aber wir wählen trotzdem {best_asset[0]}")

    print(f"🏆 Beste Aktie/Krypto: {best_asset[0]} mit Score {best_asset[1]}")
    return best_asset
else:
    print("⚠️ Keine geeignete Aktie/Krypto gefunden. Alle Scores: ", scores)
    return None, 0.0



def fetch_all_data():
    print("🚀 Starte das Laden der Aktien...")
    for ticker in stock_list:
        fetch_data_with_cache(ticker)
        wait_time = random.uniform(1, 3)  # 🔥 Verhindert API-Sperren
        print(f"⏳ Warte {wait_time:.2f} Sekunden, um API-Sperren zu vermeiden...")
        time.sleep(wait_time)
    print("✅ Alle Aktien-Daten wurden geladen!")

# 🔄 Wiederholt den Datenabruf alle 4 Stunden
def run_periodically():
    while True:
        fetch_all_data()
        print("⏳ Warte 4 Stunden, bevor die nächste Analyse startet...")
        time.sleep(4 * 60 * 60)  # 4 Stunden warten

# Hintergrund-Thread starten
thread = threading.Thread(target=run_periodically, daemon=True)
thread.start()



last_recommendation_time = 0  # Zeitstempel der letzten Empfehlung
best_stock = None  # Speichert die letzte empfohlene Aktie

@app.get("/")
def get_recommended_stock():
    """Gibt die beste Aktie basierend auf den letzten 4 Stunden zurück"""
    global last_recommendation_time, best_stock, global_scores

    current_time = time.time()
    
    # Prüfe, ob 4 Stunden vergangen sind (4 * 60 * 60 Sekunden)
    if current_time - last_recommendation_time >= 4 * 60 * 60:
        if global_scores:
            best_stock = max(global_scores, key=lambda x: x[1], default=(None, 0))
            last_recommendation_time = current_time  # Zeit aktualisieren
            print(f"🔄 Neue Empfehlung erstellt: {best_stock[0]}")
    
    if best_stock is None:
        return {"error": "Noch keine Empfehlung verfügbar"}

    return {
    "ticker": best_stock[0] if best_stock else "Keine Empfehlung",
    "indikatoren": {
        "RSI": best_stock[1] if best_stock else 0,
    }
}






