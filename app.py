<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading-App - Kaufempfehlung</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            background-color: #f8f9fa;
        }
        .container {
            width: 80%;
            margin: auto;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        .highlight {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        table {
            width: 100%;
            margin: 20px auto;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid black;
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #007bff;
            color: white;
        }
        canvas {
            width: 100%;
            height: 400px;
        }
    </style>
</head>
<body>

    <div class="container">
        <h1>🚀 Top Aktienempfehlung</h1>
        <h2 class="highlight" id="stockSymbol">Lädt...</h2>
        <p>Analyse basierend auf technischen Indikatoren & News.</p>

        <h3>📊 Technische Indikatoren</h3>
        <table>
            <tr><th>RSI</th><td id="rsi">-</td></tr>
            <tr><th>MACD</th><td id="macd">-</td></tr>
            <tr><th>SMA 50</th><td id="sma50">-</td></tr>
            <tr><th>SMA 200</th><td id="sma200">-</td></tr>
        </table>

       <section id="chart">
    <h2>Kursverlauf</h2>
    <label for="timeframe-select">Zeitraum:</label>
    <select id="timeframe-select">
        <option value="1D">1 Tag</option>
        <option value="1W">1 Woche</option>
        <option value="1M">1 Monat</option>
        <option value="6M">6 Monate</option>
        <option value="1Y">1 Jahr</option>
    </select>
    <canvas id="stock-chart"></canvas>
</section>

        <h3>📰 Wichtige Nachrichten</h3>
        <ul id="newsList">
            <li>Lädt...</li>
        </ul>
    </div>

    <script>
        document.addEventListener("DOMContentLoaded", function() {
            Chart.register(ChartFinancial, ChartDateAdapter);
            loadStockData();
        });

        async function loadStockData() {
            try {
                const response = await fetch("/recommendation");
                const data = await response.json();
                
                document.getElementById("stockSymbol").innerText = data.symbol || "Keine Empfehlung";
                document.getElementById("rsi").innerText = data.rsi !== null ? data.rsi.toFixed(2) : "N/A";
                document.getElementById("macd").innerText = data.macd !== null ? data.macd.toFixed(2) : "N/A";
                document.getElementById("sma50").innerText = data.sma_50 !== null ? data.sma_50.toFixed(2) : "N/A";
                document.getElementById("sma200").innerText = data.sma_200 !== null ? data.sma_200.toFixed(2) : "N/A";

                loadStockChart(data.history);
                
                const newsList = document.getElementById("newsList");
                newsList.innerHTML = "";
                data.news.forEach(news => {
                    const listItem = document.createElement("li");
                    listItem.innerText = `${news.title} - Bewertung: ${news.rating}/10`;
                    newsList.appendChild(listItem);
                });
            } catch (error) {
                console.error("Fehler beim Laden der API-Daten:", error);
            }
        }
    </script>

<script src="https://cdn.jsdelivr.net/npm/date-fns"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial"></script>

</body>
</html>
