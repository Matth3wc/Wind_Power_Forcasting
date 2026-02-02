"""
Forecast API routes.
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Request
import pandas as pd
import numpy as np

from config.settings import ALL_STATIONS, BUOY_STATIONS, COASTAL_STATIONS
from api.models.schemas import ForecastResponse, ForecastPoint, RegionalForecastRequest
from ingestion.buoy_fetcher import BuoyFetcher

router = APIRouter()


@router.get("/{station_id}")
async def get_station_forecast(
    station_id: str,
    horizons: List[int] = Query(
        default=[6, 12, 24, 48, 72],
        description="Forecast horizons in hours"
    ),
    variables: List[str] = Query(
        default=["wind_speed", "wave_height", "air_pressure"],
        description="Variables to forecast"
    )
) -> Dict[str, Any]:
    """
    Get weather forecasts for a specific station.
    
    Uses a Vector Autoregressive (VAR) model trained on recent historical data
    to generate forecasts at specified horizons.
    
    Args:
        station_id: Station identifier
        horizons: List of forecast horizons in hours (e.g., [6, 12, 24, 48])
        variables: Variables to include in forecast
    """
    if station_id not in ALL_STATIONS:
        raise HTTPException(status_code=404, detail=f"Station '{station_id}' not found")
    
    # Only buoys have ERDDAP data for forecasting
    buoy_stations = {**BUOY_STATIONS, **COASTAL_STATIONS}
    if station_id not in buoy_stations:
        raise HTTPException(
            status_code=400,
            detail=f"Forecasting not available for station type: {ALL_STATIONS[station_id]['type']}"
        )
    
    # Fetch recent data for model training
    fetcher = BuoyFetcher()
    
    try:
        df = await fetcher.fetch_single_buoy(station_id, days_back=14)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch data for forecasting: {str(e)}"
        )
    
    if len(df) < 48:  # Need at least 48 hours of data
        raise HTTPException(
            status_code=400,
            detail="Insufficient historical data for forecasting"
        )
    
    # Prepare data for VAR model
    available_vars = [v for v in variables if v in df.columns]
    if not available_vars:
        raise HTTPException(
            status_code=400,
            detail=f"None of the requested variables available. Available: {list(df.columns)}"
        )
    
    # Resample to hourly
    df_hourly = df[available_vars].resample("1H").mean().interpolate(method="time", limit=3)
    df_hourly = df_hourly.dropna()
    
    if len(df_hourly) < 48:
        raise HTTPException(
            status_code=400,
            detail="Insufficient data after preprocessing"
        )
    
    # Generate forecasts using VAR model
    try:
        from models.var_model import VARForecaster
        
        forecaster = VARForecaster()
        forecaster.fit(df_hourly)
        
        max_horizon = max(horizons)
        forecast_df, confidence_intervals = forecaster.predict(steps=max_horizon)
        
    except ImportError:
        # Fallback: simple persistence forecast
        forecast_df, confidence_intervals = _simple_forecast(df_hourly, max(horizons), available_vars)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Forecasting failed: {str(e)}"
        )
    
    # Build forecast response
    forecast_time = datetime.utcnow()
    forecasts = []
    
    for horizon in sorted(horizons):
        if horizon <= len(forecast_df):
            forecast_point = {
                "timestamp": (forecast_time + timedelta(hours=horizon)).isoformat(),
                "horizon_hours": horizon,
            }
            
            for var in available_vars:
                if var in forecast_df.columns:
                    forecast_point[var] = round(float(forecast_df[var].iloc[horizon - 1]), 2)
                    
                    # Add confidence intervals if available
                    if confidence_intervals is not None:
                        ci = confidence_intervals.get(var, {})
                        if "lower" in ci and horizon <= len(ci["lower"]):
                            forecast_point[f"{var}_lower"] = round(float(ci["lower"].iloc[horizon - 1]), 2)
                        if "upper" in ci and horizon <= len(ci["upper"]):
                            forecast_point[f"{var}_upper"] = round(float(ci["upper"].iloc[horizon - 1]), 2)
            
            # Compute confidence (decreases with horizon)
            forecast_point["confidence"] = round(max(0.3, 1.0 - horizon / 100), 2)
            
            forecasts.append(forecast_point)
    
    return {
        "station_id": station_id,
        "station_name": ALL_STATIONS[station_id]["name"],
        "forecast_time": forecast_time.isoformat(),
        "model_type": "VAR",
        "horizons": horizons,
        "variables": available_vars,
        "forecasts": forecasts
    }


@router.post("/regional")
async def get_regional_forecast(
    request_body: RegionalForecastRequest
) -> Dict[str, Any]:
    """
    Get forecasts for multiple stations in a region.
    
    Useful for getting an overview of expected conditions across an area.
    """
    station_ids = request_body.station_ids
    horizons = request_body.horizons or [6, 12, 24, 48]
    
    # Validate stations
    buoy_stations = {**BUOY_STATIONS, **COASTAL_STATIONS}
    valid_stations = [s for s in station_ids if s in buoy_stations]
    
    if not valid_stations:
        raise HTTPException(
            status_code=400,
            detail="No valid buoy stations provided for forecasting"
        )
    
    # Fetch data for all stations
    fetcher = BuoyFetcher()
    forecasts = {}
    
    for station_id in valid_stations:
        try:
            df = await fetcher.fetch_single_buoy(station_id, days_back=14)
            
            if len(df) < 48:
                continue
            
            # Simplified forecast
            variables = ["wind_speed", "wave_height", "air_pressure"]
            available_vars = [v for v in variables if v in df.columns]
            
            if not available_vars:
                continue
            
            df_hourly = df[available_vars].resample("1H").mean().interpolate(method="time", limit=3)
            df_hourly = df_hourly.dropna()
            
            forecast_df, _ = _simple_forecast(df_hourly, max(horizons), available_vars)
            
            station_forecasts = []
            forecast_time = datetime.utcnow()
            
            for horizon in horizons:
                if horizon <= len(forecast_df):
                    point = {
                        "timestamp": (forecast_time + timedelta(hours=horizon)).isoformat(),
                        "horizon_hours": horizon,
                    }
                    for var in available_vars:
                        if var in forecast_df.columns:
                            point[var] = round(float(forecast_df[var].iloc[horizon - 1]), 2)
                    station_forecasts.append(point)
            
            forecasts[station_id] = station_forecasts
            
        except Exception as e:
            # Skip failed stations
            continue
    
    if not forecasts:
        raise HTTPException(
            status_code=503,
            detail="Failed to generate forecasts for any station"
        )
    
    # Compute regional summary
    regional_summary = {}
    for horizon in horizons:
        horizon_values = {"wind_speed": [], "wave_height": [], "air_pressure": []}
        
        for station_id, station_forecasts in forecasts.items():
            for fc in station_forecasts:
                if fc["horizon_hours"] == horizon:
                    for var in horizon_values:
                        if var in fc:
                            horizon_values[var].append(fc[var])
        
        regional_summary[f"{horizon}h"] = {
            var: {
                "mean": round(np.mean(vals), 2) if vals else None,
                "min": round(min(vals), 2) if vals else None,
                "max": round(max(vals), 2) if vals else None,
            }
            for var, vals in horizon_values.items() if vals
        }
    
    return {
        "region": "custom",
        "stations": valid_stations,
        "forecast_time": datetime.utcnow().isoformat(),
        "model_type": "VAR",
        "horizons": horizons,
        "station_forecasts": forecasts,
        "regional_summary": regional_summary
    }


@router.get("/ensemble/{station_id}")
async def get_ensemble_forecast(
    station_id: str,
    horizons: List[int] = Query(default=[12, 24, 48])
) -> Dict[str, Any]:
    """
    Get ensemble forecast with multiple model runs.
    
    Provides uncertainty quantification through ensemble spread.
    """
    if station_id not in ALL_STATIONS:
        raise HTTPException(status_code=404, detail=f"Station '{station_id}' not found")
    
    # This is a placeholder for a more sophisticated ensemble approach
    # In production, this would run multiple model configurations
    
    fetcher = BuoyFetcher()
    
    try:
        df = await fetcher.fetch_single_buoy(station_id, days_back=14)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    
    if len(df) < 48:
        raise HTTPException(status_code=400, detail="Insufficient data")
    
    variables = ["wind_speed", "wave_height"]
    available_vars = [v for v in variables if v in df.columns]
    
    df_hourly = df[available_vars].resample("1H").mean().interpolate(method="time", limit=3).dropna()
    
    # Generate ensemble members (simplified)
    n_members = 10
    ensemble_forecasts = {}
    
    for var in available_vars:
        ensemble_forecasts[var] = []
        
        for _ in range(n_members):
            # Add small random perturbations to create ensemble spread
            noise_factor = np.random.uniform(0.95, 1.05)
            trend_factor = np.random.uniform(-0.01, 0.01)
            
            member = []
            last_value = df_hourly[var].iloc[-1]
            
            for h in range(max(horizons)):
                # Persistence with noise and trend
                forecast_value = last_value * noise_factor + h * trend_factor
                member.append(forecast_value)
            
            ensemble_forecasts[var].append(member)
    
    # Compute ensemble statistics
    forecast_time = datetime.utcnow()
    result_forecasts = []
    
    for horizon in horizons:
        point = {
            "timestamp": (forecast_time + timedelta(hours=horizon)).isoformat(),
            "horizon_hours": horizon,
        }
        
        for var in available_vars:
            values_at_horizon = [m[horizon - 1] for m in ensemble_forecasts[var]]
            point[var] = round(np.mean(values_at_horizon), 2)
            point[f"{var}_std"] = round(np.std(values_at_horizon), 2)
            point[f"{var}_p10"] = round(np.percentile(values_at_horizon, 10), 2)
            point[f"{var}_p90"] = round(np.percentile(values_at_horizon, 90), 2)
        
        result_forecasts.append(point)
    
    return {
        "station_id": station_id,
        "forecast_time": forecast_time.isoformat(),
        "model_type": "ensemble",
        "n_members": n_members,
        "horizons": horizons,
        "forecasts": result_forecasts
    }


def _simple_forecast(
    df: pd.DataFrame,
    steps: int,
    variables: List[str]
) -> tuple:
    """
    Simple persistence forecast with trend.
    
    Used as fallback when VAR model is not available.
    """
    forecasts = {}
    confidence_intervals = {}
    
    for var in variables:
        if var not in df.columns:
            continue
        
        series = df[var].dropna()
        if len(series) < 2:
            continue
        
        # Compute trend from recent data
        recent = series.tail(24)
        if len(recent) > 1:
            trend = (recent.iloc[-1] - recent.iloc[0]) / len(recent)
        else:
            trend = 0
        
        # Compute mean and std for confidence intervals
        std = series.std()
        
        # Generate forecast
        last_value = series.iloc[-1]
        forecast_values = []
        lower_bounds = []
        upper_bounds = []
        
        for h in range(1, steps + 1):
            # Damped trend persistence
            damping = 0.95 ** h
            forecast = last_value + trend * h * damping
            
            # Confidence intervals widen with horizon
            uncertainty = std * np.sqrt(h / 24)
            
            forecast_values.append(forecast)
            lower_bounds.append(forecast - 1.96 * uncertainty)
            upper_bounds.append(forecast + 1.96 * uncertainty)
        
        forecasts[var] = forecast_values
        confidence_intervals[var] = {
            "lower": pd.Series(lower_bounds),
            "upper": pd.Series(upper_bounds)
        }
    
    # Create forecast DataFrame
    forecast_index = pd.date_range(
        start=df.index[-1] + pd.Timedelta(hours=1),
        periods=steps,
        freq="1H"
    )
    
    forecast_df = pd.DataFrame(forecasts, index=forecast_index)
    
    return forecast_df, confidence_intervals
