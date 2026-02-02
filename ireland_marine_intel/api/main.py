"""
FastAPI main application for Ireland Marine Weather Intelligence Platform.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from config.settings import get_settings
from api.routes import stations, weather, forecasts, analysis
from ingestion.scheduler import RealTimeDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global data manager
data_manager: RealTimeDataManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global data_manager
    
    logger.info("Starting Ireland Marine Weather Intelligence Platform...")
    
    # Initialize data manager
    data_manager = RealTimeDataManager()
    app.state.data_manager = data_manager
    
    # Start data collection scheduler
    data_manager.start()
    
    # Perform initial data fetch
    logger.info("Performing initial data fetch...")
    try:
        await data_manager.scheduler.fetch_all_data()
    except Exception as e:
        logger.error(f"Initial data fetch failed: {e}")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    data_manager.stop()
    logger.info("Application shutdown complete")


# Create FastAPI app
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    Ireland Live Marine Weather Intelligence Platform API.
    
    Provides real-time weather data from Irish marine buoys and coastal stations,
    forecasting using VAR models, and network analysis of weather patterns.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    stations.router,
    prefix="/api/stations",
    tags=["Stations"]
)

app.include_router(
    weather.router,
    prefix="/api/weather",
    tags=["Weather"]
)

app.include_router(
    forecasts.router,
    prefix="/api/forecasts",
    tags=["Forecasts"]
)

app.include_router(
    analysis.router,
    prefix="/api/analysis",
    tags=["Analysis"]
)


# WebSocket connection manager
class ConnectionManager:
    """Manage WebSocket connections for live updates."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


@app.websocket("/ws/live")
async def websocket_live_data(websocket: WebSocket):
    """
    WebSocket endpoint for live data updates.
    
    Clients receive real-time updates when new data is fetched.
    """
    await manager.connect(websocket)
    
    # Subscribe to data updates
    queue = data_manager.subscribe()
    
    try:
        # Send current data immediately
        current_data = data_manager.get_latest_readings()
        await websocket.send_json({
            "type": "initial_data",
            "timestamp": datetime.utcnow().isoformat(),
            "data": current_data
        })
        
        # Listen for updates
        while True:
            try:
                # Wait for updates from data manager
                update = await asyncio.wait_for(queue.get(), timeout=30)
                await websocket.send_json(update)
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        data_manager.unsubscribe(queue)
        manager.disconnect(websocket)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - serve the frontend directly."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ireland Marine Weather API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #1f77b4; }
            a { color: #1f77b4; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
            code { background: #e0e0e0; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>🌊 Ireland Marine Weather Intelligence API</h1>
        <p>Real-time weather data from Irish marine buoys and coastal stations.</p>
        
        <h2>Quick Links</h2>
        <ul>
            <li><a href="/docs">API Documentation (Swagger)</a></li>
            <li><a href="/redoc">API Documentation (ReDoc)</a></li>
            <li><a href="/api/stations">All Stations</a></li>
            <li><a href="/api/weather/latest">Latest Weather Data</a></li>
        </ul>
        
        <h2>Key Endpoints</h2>
        <div class="endpoint">
            <strong>GET /api/stations</strong> - List all monitoring stations
        </div>
        <div class="endpoint">
            <strong>GET /api/weather/latest</strong> - Get latest readings from all stations
        </div>
        <div class="endpoint">
            <strong>GET /api/weather/{station_id}/history</strong> - Historical data for a station
        </div>
        <div class="endpoint">
            <strong>GET /api/forecasts/{station_id}</strong> - Weather forecasts
        </div>
        <div class="endpoint">
            <strong>GET /api/analysis/network</strong> - Network analysis results
        </div>
        <div class="endpoint">
            <strong>WS /ws/live</strong> - WebSocket for real-time updates
        </div>
        
        <h2>Frontend</h2>
        <p>Access the interactive map at <a href="/frontend/">/frontend/</a></p>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
        "scheduler_status": data_manager.scheduler.get_status() if data_manager else None
    }


@app.get("/api/status")
async def api_status():
    """Get API and data collection status."""
    if not data_manager:
        return {"status": "initializing"}
    
    return {
        "status": "running",
        "scheduler": data_manager.scheduler.get_status(),
        "stations_with_data": len(data_manager.get_latest_readings()),
        "last_update": data_manager.scheduler._last_data.get("timestamp", None),
    }


# Mount static files for frontend
import os
from pathlib import Path
from fastapi.responses import FileResponse

# Get the base directory (where api/ is located)
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Serve frontend static files
@app.get("/app")
@app.get("/app/")
async def serve_frontend():
    """Serve the frontend application."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"error": "Frontend not found"}

# Mount static directories
try:
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    logger.info(f"Mounted frontend from {FRONTEND_DIR}")
except Exception as e:
    logger.warning(f"Could not mount frontend static files: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
