"""
Pydantic schemas for API request/response models.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class StationMetadata(BaseModel):
    """Station metadata."""
    station_id: str = Field(..., description="Unique station identifier")
    name: str = Field(..., description="Station name")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    station_type: str = Field(..., description="Type: offshore_buoy, coastal_buoy, lighthouse, synoptic")
    depth_m: Optional[float] = Field(None, description="Water depth in meters")
    description: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "station_id": "M5",
                "name": "M5 Buoy",
                "latitude": 51.69,
                "longitude": -6.704,
                "station_type": "offshore_buoy",
                "depth_m": 90,
                "description": "Celtic Sea weather buoy"
            }
        }


class WeatherReading(BaseModel):
    """Single weather reading."""
    timestamp: datetime
    station_id: str
    wind_speed: Optional[float] = Field(None, description="Wind speed in knots")
    wind_direction: Optional[float] = Field(None, description="Wind direction in degrees")
    wind_gust: Optional[float] = Field(None, description="Wind gust in knots")
    wave_height: Optional[float] = Field(None, description="Significant wave height in meters")
    wave_height_max: Optional[float] = Field(None, description="Maximum wave height in meters")
    wave_period: Optional[float] = Field(None, description="Peak wave period in seconds")
    wave_direction: Optional[float] = Field(None, description="Wave direction in degrees")
    air_temperature: Optional[float] = Field(None, description="Air temperature in Celsius")
    sea_temperature: Optional[float] = Field(None, description="Sea surface temperature in Celsius")
    air_pressure: Optional[float] = Field(None, description="Atmospheric pressure in millibars")
    visibility: Optional[float] = Field(None, description="Visibility in km")
    humidity: Optional[float] = Field(None, description="Relative humidity %")
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2026-02-02T12:00:00Z",
                "station_id": "M5",
                "wind_speed": 25.3,
                "wind_direction": 240,
                "wave_height": 3.2,
                "wave_period": 8.5,
                "air_temperature": 9.5,
                "sea_temperature": 10.2,
                "air_pressure": 1013.5
            }
        }


class LatestReadingsResponse(BaseModel):
    """Response for latest readings endpoint."""
    timestamp: datetime
    stations: Dict[str, WeatherReading]
    count: int


class HistoricalDataResponse(BaseModel):
    """Response for historical data endpoint."""
    station_id: str
    start_time: datetime
    end_time: datetime
    count: int
    data: List[WeatherReading]


class ForecastPoint(BaseModel):
    """Single forecast point."""
    timestamp: datetime
    horizon_hours: int = Field(..., description="Hours ahead from forecast time")
    wind_speed: Optional[float] = None
    wind_speed_lower: Optional[float] = Field(None, description="Lower confidence bound")
    wind_speed_upper: Optional[float] = Field(None, description="Upper confidence bound")
    wave_height: Optional[float] = None
    wave_height_lower: Optional[float] = None
    wave_height_upper: Optional[float] = None
    air_pressure: Optional[float] = None
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Forecast confidence")


class ForecastResponse(BaseModel):
    """Response for forecast endpoint."""
    station_id: str
    forecast_time: datetime = Field(..., description="When forecast was generated")
    model_type: str = Field(..., description="Model used (e.g., 'VAR')")
    horizons: List[int] = Field(..., description="Forecast horizons in hours")
    forecasts: List[ForecastPoint]


class RegionalForecastResponse(BaseModel):
    """Response for regional forecast endpoint."""
    region: str
    stations: List[str]
    forecast_time: datetime
    model_type: str
    forecasts: Dict[str, List[ForecastPoint]]


class NetworkNode(BaseModel):
    """Node in the station network."""
    station_id: str
    latitude: float
    longitude: float
    station_type: str
    cluster_id: Optional[int] = None
    centrality: Optional[float] = None


class NetworkEdge(BaseModel):
    """Edge in the station network."""
    source: str
    target: str
    weight: float
    edge_type: str = Field(..., description="Type: correlation, distance, flow")


class NetworkAnalysisResponse(BaseModel):
    """Response for network analysis endpoint."""
    timestamp: datetime
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]
    clusters: Dict[int, List[str]]
    statistics: Dict[str, Any]


class FlowPattern(BaseModel):
    """Weather flow pattern between stations."""
    source_station: str
    target_station: str
    lag_hours: float
    correlation: float
    direction: str = Field(..., description="Predominant flow direction")
    variables: List[str] = Field(..., description="Variables involved")


class FlowAnalysisResponse(BaseModel):
    """Response for flow analysis endpoint."""
    timestamp: datetime
    patterns: List[FlowPattern]
    dominant_flow_direction: str
    flow_speed_estimate: Optional[float] = Field(None, description="Estimated flow speed in km/h")


class BathymetryInfo(BaseModel):
    """Bathymetry information for a station."""
    station_id: str
    depth_m: float
    depth_category: str
    shelf_position: str
    wave_exposure: Dict[str, str]


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Request models
class HistoricalDataRequest(BaseModel):
    """Request parameters for historical data."""
    days_back: int = Field(default=7, ge=1, le=365)
    resample_freq: Optional[str] = Field(default="1H", description="Resample frequency (e.g., '1H', '30T')")


class ForecastRequest(BaseModel):
    """Request parameters for forecasts."""
    horizons: Optional[List[int]] = Field(
        default=[6, 12, 24, 48, 72],
        description="Forecast horizons in hours"
    )
    variables: Optional[List[str]] = Field(
        default=["wind_speed", "wave_height", "air_pressure"],
        description="Variables to forecast"
    )


class RegionalForecastRequest(BaseModel):
    """Request for regional forecast."""
    station_ids: List[str] = Field(..., description="Station IDs in the region")
    horizons: Optional[List[int]] = Field(default=[6, 12, 24, 48])
