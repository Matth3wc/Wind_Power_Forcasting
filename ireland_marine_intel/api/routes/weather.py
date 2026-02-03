"""
Weather data API routes.
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Request
import pandas as pd

from config.settings import ALL_STATIONS, BUOY_STATIONS, COASTAL_STATIONS, LIGHTHOUSE_STATIONS
from api.models.schemas import WeatherReading, LatestReadingsResponse, HistoricalDataResponse
from ingestion.buoy_fetcher import BuoyFetcher
from ingestion.lighthouse_fetcher import LighthouseFetcher

router = APIRouter()


def is_buoy_station(station_id: str) -> bool:
    """Check if station is a buoy (ERDDAP data source)."""
    return station_id in BUOY_STATIONS or station_id in COASTAL_STATIONS


def is_lighthouse_station(station_id: str) -> bool:
    """Check if station is a Met Éireann AWS station."""
    return station_id in LIGHTHOUSE_STATIONS


def df_row_to_weather_reading(station_id: str, row: pd.Series, timestamp: datetime) -> WeatherReading:
    """Convert a DataFrame row to WeatherReading."""
    return WeatherReading(
        timestamp=timestamp,
        station_id=station_id,
        wind_speed=row.get("wind_speed"),
        wind_direction=row.get("wind_direction"),
        wind_gust=row.get("wind_gust"),
        wave_height=row.get("wave_height"),
        wave_height_max=row.get("wave_height_max"),
        wave_period=row.get("wave_period"),
        wave_direction=row.get("wave_direction"),
        air_temperature=row.get("air_temperature"),
        sea_temperature=row.get("sea_temperature"),
        air_pressure=row.get("air_pressure"),
        visibility=row.get("visibility"),
        humidity=row.get("humidity"),
    )


@router.get("/latest")
async def get_latest_readings(request: Request) -> Dict[str, Any]:
    """
    Get the most recent reading from all stations.
    
    Returns the latest weather data from each active station.
    """
    data_manager = request.app.state.data_manager
    
    if not data_manager:
        raise HTTPException(status_code=503, detail="Data manager not initialized")
    
    latest = data_manager.get_latest_readings()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "count": len(latest),
        "stations": latest
    }


@router.get("/{station_id}/latest", response_model=WeatherReading)
async def get_station_latest(station_id: str, request: Request) -> WeatherReading:
    """
    Get the most recent reading for a specific station.
    
    Args:
        station_id: Station identifier
    """
    if station_id not in ALL_STATIONS:
        raise HTTPException(status_code=404, detail=f"Station '{station_id}' not found")
    
    data_manager = request.app.state.data_manager
    latest = data_manager.get_latest_readings()
    
    if station_id not in latest:
        raise HTTPException(
            status_code=404,
            detail=f"No recent data for station '{station_id}'"
        )
    
    reading_data = latest[station_id]
    
    return WeatherReading(
        timestamp=datetime.fromisoformat(reading_data.get("timestamp", datetime.utcnow().isoformat())),
        station_id=station_id,
        **{k: v for k, v in reading_data.items() if k != "timestamp"}
    )


@router.get("/{station_id}/history")
async def get_station_history(
    station_id: str,
    days_back: int = Query(7, ge=1, le=365, description="Number of days of history"),
    resample_freq: Optional[str] = Query("1h", description="Resample frequency (e.g., '1h', '30min', 'D')")
) -> Dict[str, Any]:
    """
    Get historical weather data for a station.
    
    Args:
        station_id: Station identifier
        days_back: Number of days of historical data
        resample_freq: Optional resampling frequency
    """
    if station_id not in ALL_STATIONS:
        raise HTTPException(status_code=404, detail=f"Station '{station_id}' not found")
    
    try:
        # Use appropriate fetcher based on station type
        if is_buoy_station(station_id):
            fetcher = BuoyFetcher()
            df = await fetcher.fetch_single_buoy(station_id, days_back=days_back)
        elif is_lighthouse_station(station_id):
            # Met Éireann stations - limit to 7 days max due to API structure
            actual_days = min(days_back, 7)
            async with LighthouseFetcher() as fetcher:
                df = await fetcher.fetch_met_eireann_data(station_id, days_back=actual_days)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown station type for '{station_id}'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch data for station '{station_id}': {str(e)}"
        )
    
    if df.empty:
        return {
            "station_id": station_id,
            "start_time": None,
            "end_time": None,
            "count": 0,
            "data": []
        }
    
    # Optionally resample
    if resample_freq:
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df = df[numeric_cols].resample(resample_freq).mean()
        df = df.dropna(how="all")
    
    # Convert to list of readings
    data = []
    for timestamp, row in df.iterrows():
        reading = {
            "timestamp": timestamp.isoformat(),
            "station_id": station_id,
        }
        for col in df.columns:
            if pd.notna(row[col]):
                reading[col] = round(float(row[col]), 2)
        data.append(reading)
    
    return {
        "station_id": station_id,
        "start_time": df.index[0].isoformat() if len(df) > 0 else None,
        "end_time": df.index[-1].isoformat() if len(df) > 0 else None,
        "count": len(data),
        "data": data
    }


@router.get("/{station_id}/statistics")
async def get_station_statistics(
    station_id: str,
    days_back: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    variable: Optional[str] = Query(None, description="Specific variable to analyze")
) -> Dict[str, Any]:
    """
    Get statistical summary for a station's weather data.
    
    Returns mean, std, min, max, and percentiles for each variable.
    """
    if station_id not in ALL_STATIONS:
        raise HTTPException(status_code=404, detail=f"Station '{station_id}' not found")
    
    try:
        # Use appropriate fetcher based on station type
        if is_buoy_station(station_id):
            fetcher = BuoyFetcher()
            df = await fetcher.fetch_single_buoy(station_id, days_back=days_back)
        elif is_lighthouse_station(station_id):
            # Met Éireann stations - limit to 7 days max
            actual_days = min(days_back, 7)
            async with LighthouseFetcher() as fetcher:
                df = await fetcher.fetch_met_eireann_data(station_id, days_back=actual_days)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown station type for '{station_id}'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch data: {str(e)}"
        )
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available for this period")
    
    # Filter to specific variable if requested
    if variable and variable in df.columns:
        df = df[[variable]]
    
    # Compute statistics
    stats = {}
    for col in df.select_dtypes(include=["number"]).columns:
        series = df[col].dropna()
        if len(series) > 0:
            stats[col] = {
                "count": int(len(series)),
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "percentile_25": round(float(series.quantile(0.25)), 2),
                "percentile_50": round(float(series.quantile(0.50)), 2),
                "percentile_75": round(float(series.quantile(0.75)), 2),
            }
    
    return {
        "station_id": station_id,
        "period_days": days_back,
        "start_time": df.index[0].isoformat() if len(df) > 0 else None,
        "end_time": df.index[-1].isoformat() if len(df) > 0 else None,
        "statistics": stats
    }


@router.get("/regional/summary")
async def get_regional_summary(
    station_ids: List[str] = Query(..., description="List of station IDs"),
    request: Request = None
) -> Dict[str, Any]:
    """
    Get a summary of current conditions across multiple stations.
    
    Useful for getting a quick overview of a region.
    """
    # Validate stations
    for sid in station_ids:
        if sid not in ALL_STATIONS:
            raise HTTPException(status_code=404, detail=f"Station '{sid}' not found")
    
    data_manager = request.app.state.data_manager
    latest = data_manager.get_latest_readings()
    
    # Filter to requested stations
    region_data = {sid: latest.get(sid) for sid in station_ids if sid in latest}
    
    if not region_data:
        return {
            "stations": station_ids,
            "count": 0,
            "summary": None
        }
    
    # Compute regional statistics
    variables = ["wind_speed", "wave_height", "air_temperature", "air_pressure"]
    summary = {}
    
    for var in variables:
        values = [d.get(var) for d in region_data.values() if d and d.get(var) is not None]
        if values:
            summary[var] = {
                "mean": round(sum(values) / len(values), 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "range": round(max(values) - min(values), 2),
            }
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "stations": station_ids,
        "active_stations": len(region_data),
        "summary": summary,
        "station_data": region_data
    }
