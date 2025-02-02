import os
import requests
from fastapi import FastAPI
from dotenv import load_dotenv

# Lade Umgebungsvariablen aus .env
load_dotenv()

app = FastAPI()

# 🔹 API-Daten aus Umgebungsvariablen
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

# 📌 Endpunkt für Nachrichten-Suche
@app.get("/get_news")
def get_news(query: str):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query + " stock news",
        "num": 5,  # Anzahl der Ergebnisse
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        news_results = []
        
        for item in data.get("items", []):
            news_results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            })
        
        return {"news": news_results}
    else:
        return {"error": "Fehler bei der Google-Suche"}
