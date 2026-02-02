# Ireland Live Marine Weather Intelligence Platform

A comprehensive full-stack application for real-time marine weather monitoring, forecasting, and analysis around Ireland.

## 🌊 Features

- **Live Data Ingestion**: Real-time data from Irish weather buoys and coastal lighthouses via ERDDAP
- **Interactive Map Visualization**: Live weather state of Ireland with color-coded sea state and wind vectors
- **VAR Forecasting**: Vector Autoregressive models for short-to-medium term weather prediction
- **Network/Mesh Analysis**: Identify weather flow pathways and correlated station clusters
- **Bathymetry Integration**: Seabed depth analysis for understanding wave patterns

## 📁 Project Structure

```
ireland_marine_intel/
├── ingestion/          # Data collection from ERDDAP and other sources
│   ├── erddap_client.py
│   ├── buoy_fetcher.py
│   ├── lighthouse_fetcher.py
│   ├── bathymetry_fetcher.py
│   └── scheduler.py
├── api/                # FastAPI backend
│   ├── main.py
│   ├── routes/
│   │   ├── stations.py
│   │   ├── weather.py
│   │   ├── forecasts.py
│   │   └── analysis.py
│   ├── models/
│   │   └── schemas.py
│   └── database/
│       ├── connection.py
│       └── models.py
├── models/             # ML/Statistical models
│   ├── var_model.py
│   ├── forecaster.py
│   └── model_utils.py
├── analysis/           # Network and mesh analysis
│   ├── network_builder.py
│   ├── flow_analysis.py
│   ├── correlation_network.py
│   └── bathymetry_analysis.py
├── frontend/           # HTML/JS frontend
│   ├── index.html
│   ├── css/
│   │   └── styles.css
│   └── js/
│       ├── app.js
│       ├── map.js
│       ├── network.js
│       ├── charts.js
│       └── websocket.js
├── config/             # Configuration files
│   └── settings.py
├── tests/              # Unit tests
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL with PostGIS (optional, SQLite works for development)
- Node.js (optional, for frontend development)

### Installation

1. Create and activate a virtual environment:
```bash
cd ireland_marine_intel
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the API server:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

4. Open the frontend:
```bash
# Open frontend/index.html in your browser, or
python -m http.server 3000 --directory frontend
```

5. Access the application:
- API Documentation: http://localhost:8000/docs
- Frontend: http://localhost:3000

## 📊 Data Sources

### Primary Sources (ERDDAP)
- **Irish Weather Buoy Network**: M2, M3, M4, M5, M6
- **Coastal Buoys**: IL1, IL2, IL3, IL4
- **Met Éireann Synoptic Stations**: Coastal weather stations

### Variables Collected
- Wind speed and direction
- Wave height, period, and direction
- Sea surface temperature
- Air temperature and pressure
- Visibility
- Precipitation

## 🔮 Forecasting

The VAR (Vector Autoregressive) model provides:
- **Single-site forecasts**: Predict conditions at a specific buoy/lighthouse
- **Regional forecasts**: Predict conditions across a region of stations
- **Horizons**: 6h, 12h, 24h, 48h, 72h forecasts

## 🕸️ Network Analysis

- **Correlation Networks**: Identify stations with similar weather patterns
- **Community Detection**: Find clusters of related stations
- **Flow Analysis**: Track propagation of weather systems
- **Bathymetry Correlation**: Relate seabed features to weather patterns

## 🛠️ API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/stations` | List all stations with metadata |
| `GET /api/stations/{id}` | Get station details |
| `GET /api/weather/latest` | Get latest readings from all stations |
| `GET /api/weather/{station_id}/history` | Get historical data |
| `GET /api/forecasts/{station_id}` | Get forecasts for a station |
| `GET /api/forecasts/regional` | Get regional forecasts |
| `GET /api/analysis/network` | Get network analysis results |
| `GET /api/analysis/flows` | Get weather flow analysis |
| `WS /ws/live` | WebSocket for live updates |

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please read the contributing guidelines first.
