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
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK.B", "JPM",
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
              "Close": float(data["Close"].iloc[-1].squeeze()),  # squeeze() stellt sicher, dass nur ein Wert bleibt
              "RSI": float(ta.momentum.RSIIndicator(data["Close"]).rsi().dropna().values[-1]) if not ta.momentum.RSIIndicator(data["Close"]).rsi().dropna().empty else None,
              "MACD": float(ta.trend.MACD(data["Close"]).macd().dropna().values[-1]) if not ta.trend.MACD(data["Close"]).macd().dropna().empty else None,
              "SMA50": float(ta.trend.SMAIndicator(data["Close"], window=50).sma_indicator().dropna().values[-1]) if not ta.trend.SMAIndicator(data["Close"], window=50).sma_indicator().dropna().empty else None,
              "SMA200": float(ta.trend.SMAIndicator(data["Close"], window=200).sma_indicator().dropna().values[-1]) if not ta.trend.SMAIndicator(data["Close"], window=200).sma_indicator().dropna().empty else None,
             })


            print(f"📈 Berechnete Indikatoren für {ticker}: {df.to_dict(orient='records')}")  # Debugging
            
            # Vorhersage mit KI-Modell
            prediction = model.predict(df)[0] if model else 0
            sentiment = get_news_sentiment(ticker)
            final_score = prediction + sentiment

            print(f"🤖 KI-Einschätzung für {ticker}: Prediction={prediction}, Sentiment={sentiment}, Final Score={final_score}")  # Debugging

            scores.append((ticker, final_score))
            global_scores.append((ticker, final_score))  # Hinzufügen zu global_scores


        except Exception as e:
            print(f"❌ Fehler bei {ticker}: {e}")
            continue
            print(f"🔎 DEBUG: scores = {scores}")  # Gibt alle berechneten Aktien aus
            print(f"🔎 DEBUG: best_asset = {best_asset}")  # Gibt die beste Aktie aus


    # Falls Scores existieren, beste Aktie/Krypto auswählen
    if scores:
        best_asset = max(scores, key=lambda x: x[1])
        print(f"🏆 Beste Aktie/Krypto: {best_asset[0]} mit Score {best_asset[1]}")  # Debugging
        return best_asset
    else:
        print("⚠️ Keine geeignete Aktie/Krypto gefunden.")
        return None, 0.0




# API-Route für die empfohlene Aktie/Krypto
from fastapi.responses import FileResponse
import os

from fastapi.responses import FileResponse
from fastapi.requests import Request

@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def serve_index(request: Request):
    if "text/html" in request.headers.get("accept", ""):
        return FileResponse("static/index.html")
    
    try:
        best_asset, score = select_best_asset()
    except Exception as e:
        print(f"❌ Fehler in select_best_asset: {e}")
        return {"error": "Interner Fehler bei der Auswahl des besten Assets"}
    
    return {"best_asset": best_asset, "score": score, "full_list": stock_list}  # ✅ Fehlende Rückgabe hinzugefügt



@app.get("/api/empfohlene_aktie")
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

@app.get("/debug_model")
def debug_model():
    if model is None:
        return {"error": "Modell wurde nicht geladen!"}
    
    try:
        test_data = pd.DataFrame([[150, 55, 0.5, 200, 210]], columns=["close", "RSI", "MACD", "SMA50", "SMA200"])
        prediction = int(model.predict(test_data)[0])
        return {"test_prediction": prediction}
    except Exception as e:
        return {"error": str(e)}
