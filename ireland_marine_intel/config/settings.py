"""
Configuration settings for the Ireland Marine Weather Intelligence Platform.
"""
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "Ireland Marine Weather Intelligence"
    app_version: str = "1.0.0"
    debug: bool = False
    reload: bool = False
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api"
    cors_origins: str = "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000"
    
    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/marine_weather.db",
        description="Database connection URL"
    )
    
    # ERDDAP Sources
    marine_ie_erddap_url: str = "https://erddap.marine.ie/erddap"
    marine_ie_dataset_id: str = "IWBNetwork"
    met_eireann_erddap_url: str = "https://erddap.met.ie/erddap"
    
    # Data Collection Settings
    fetch_interval_minutes: int = 15  # How often to fetch new data
    history_days_default: int = 30  # Default historical data window
    max_concurrent_requests: int = 5
    request_timeout_seconds: int = 30
    buoy_update_interval: int = 30  # minutes
    lighthouse_update_interval: int = 60  # minutes
    forecast_update_interval: int = 360  # minutes
    
    # Caching
    cache_ttl_seconds: int = 300  # 5 minutes
    enable_caching: bool = True
    
    # VAR Model Settings
    var_max_lags: int = 24  # Maximum lags to consider (24 hours)
    var_information_criterion: str = "aic"  # 'aic', 'bic', 'hqic'
    var_forecast_horizons: list[int] = [6, 12, 24, 48, 72]  # Hours ahead
    forecast_horizons: str = "6,12,24,48,72"  # Comma-separated for env var
    var_retrain_interval_hours: int = 6
    
    # Network Analysis
    correlation_threshold: float = 0.5
    distance_threshold_km: float = 200.0
    
    # WebSocket
    ws_heartbeat_interval: int = 30
    ws_max_connections: int = 100
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # Ignore extra env vars not defined in the model
    }


# Station Metadata
BUOY_STATIONS = {
    # Irish Weather Buoy Network (offshore)
    "M2": {
        "name": "M2 Buoy",
        "lat": 53.4800,
        "lon": -5.4250,
        "type": "offshore_buoy",
        "depth_m": None,
        "description": "Irish Sea weather buoy"
    },
    "M3": {
        "name": "M3 Buoy",
        "lat": 51.2166,
        "lon": -10.5500,
        "type": "offshore_buoy",
        "depth_m": None,
        "description": "Southwest Atlantic buoy"
    },
    "M4": {
        "name": "M4 Buoy",
        "lat": 55.0000,
        "lon": -10.0000,
        "type": "offshore_buoy",
        "depth_m": None,
        "description": "Northwest Atlantic buoy"
    },
    "M5": {
        "name": "M5 Buoy",
        "lat": 51.6900,
        "lon": -6.7040,
        "type": "offshore_buoy",
        "depth_m": None,
        "description": "Celtic Sea weather buoy"
    },
    "M6": {
        "name": "M6 Buoy",
        "lat": 53.0748,
        "lon": -15.8814,
        "type": "offshore_buoy",
        "depth_m": None,
        "description": "Deep Atlantic buoy"
    },
}

COASTAL_STATIONS = {
    # Irish Lights / Coastal network
    "IL1": {
        "name": "IL1 Buoy (Ballybunnion)",
        "lat": 52.5420,
        "lon": -9.7820,
        "type": "coastal_buoy",
        "depth_m": 30,
        "description": "Coastal buoy near Ballybunnion"
    },
    "IL2": {
        "name": "IL2 Buoy (Finnis)",
        "lat": 53.0470,
        "lon": -9.4850,
        "type": "coastal_buoy",
        "depth_m": 25,
        "description": "Coastal buoy at Finnis Rock"
    },
    "IL3": {
        "name": "IL3 Buoy (South Hunter)",
        "lat": 54.8780,
        "lon": -5.7550,
        "type": "coastal_buoy",
        "depth_m": 35,
        "description": "Northern coastal buoy"
    },
    "IL4": {
        "name": "IL4 Buoy (Splaugh)",
        "lat": 52.2390,
        "lon": -6.2800,
        "type": "coastal_buoy",
        "depth_m": 20,
        "description": "Southeast coastal buoy"
    },
}

# Coastal lighthouses and weather stations
LIGHTHOUSE_STATIONS = {
    "MLIN": {
        "name": "Mace Head",
        "lat": 53.3269,
        "lon": -9.8989,
        "type": "lighthouse",
        "description": "Atmospheric research station"
    },
    "BELMULLET": {
        "name": "Belmullet",
        "lat": 54.2275,
        "lon": -10.0078,
        "type": "synoptic",
        "description": "Met Éireann synoptic station"
    },
    "VALENTIA": {
        "name": "Valentia Observatory",
        "lat": 51.9381,
        "lon": -10.2436,
        "type": "observatory",
        "description": "Historical weather observatory"
    },
    "MALIN_HEAD": {
        "name": "Malin Head",
        "lat": 55.3719,
        "lon": -7.3389,
        "type": "synoptic",
        "description": "Ireland's most northerly point"
    },
    "ROCHES_POINT": {
        "name": "Roches Point",
        "lat": 51.7925,
        "lon": -8.2492,
        "type": "lighthouse",
        "description": "Cork Harbour lighthouse"
    },
    "HOOK_HEAD": {
        "name": "Hook Head",
        "lat": 52.1239,
        "lon": -6.9294,
        "type": "lighthouse",
        "description": "One of the oldest lighthouses"
    },
    "LOOP_HEAD": {
        "name": "Loop Head",
        "lat": 52.5608,
        "lon": -9.9306,
        "type": "lighthouse",
        "description": "Clare coast lighthouse"
    },
}

# Combine all stations
ALL_STATIONS = {**BUOY_STATIONS, **COASTAL_STATIONS, **LIGHTHOUSE_STATIONS}

# Weather variables to collect
WEATHER_VARIABLES = {
    "wind_speed": {
        "unit": "knots",
        "description": "Wind speed",
        "min_valid": 0,
        "max_valid": 150
    },
    "wind_direction": {
        "unit": "degrees",
        "description": "Wind direction (from)",
        "min_valid": 0,
        "max_valid": 360
    },
    "wave_height": {
        "unit": "meters",
        "description": "Significant wave height",
        "min_valid": 0,
        "max_valid": 30
    },
    "wave_period": {
        "unit": "seconds",
        "description": "Peak wave period",
        "min_valid": 0,
        "max_valid": 30
    },
    "wave_direction": {
        "unit": "degrees",
        "description": "Wave direction (from)",
        "min_valid": 0,
        "max_valid": 360
    },
    "sea_temperature": {
        "unit": "celsius",
        "description": "Sea surface temperature",
        "min_valid": -5,
        "max_valid": 35
    },
    "air_temperature": {
        "unit": "celsius",
        "description": "Air temperature",
        "min_valid": -30,
        "max_valid": 50
    },
    "air_pressure": {
        "unit": "millibars",
        "description": "Atmospheric pressure",
        "min_valid": 900,
        "max_valid": 1100
    },
    "visibility": {
        "unit": "km",
        "description": "Horizontal visibility",
        "min_valid": 0,
        "max_valid": 100
    },
    "humidity": {
        "unit": "percent",
        "description": "Relative humidity",
        "min_valid": 0,
        "max_valid": 100
    },
}


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
