"""
Forecasting utilities and model management.
"""
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from .var_model import VARForecaster, MultiStationForecaster, SimplePersistenceForecaster, VARModelConfig

logger = logging.getLogger(__name__)


class ForecastManager:
    """
    Manages forecasting models for the weather intelligence platform.
    
    Handles model training, caching, and forecast generation.
    """
    
    def __init__(self, model_dir: str = "models/saved"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.station_models: Dict[str, VARForecaster] = {}
        self.regional_model: Optional[MultiStationForecaster] = None
        self.last_training_times: Dict[str, datetime] = {}
        
        # Configuration
        self.retrain_interval_hours = 6
        self.min_training_samples = 168  # 1 week of hourly data
    
    def needs_retraining(self, station_id: str) -> bool:
        """Check if a station model needs retraining."""
        if station_id not in self.last_training_times:
            return True
        
        elapsed = datetime.utcnow() - self.last_training_times[station_id]
        return elapsed > timedelta(hours=self.retrain_interval_hours)
    
    async def train_station_model(
        self,
        station_id: str,
        data: pd.DataFrame,
        variables: Optional[List[str]] = None,
        force: bool = False
    ) -> bool:
        """
        Train or update a VAR model for a station.
        
        Args:
            station_id: Station identifier
            data: Historical data DataFrame
            variables: Variables to include in model
            force: Force retraining even if not needed
        
        Returns:
            True if training succeeded
        """
        if not force and not self.needs_retraining(station_id):
            logger.debug(f"Station {station_id} model is up to date")
            return True
        
        # Prepare data
        if variables:
            data = data[variables].copy()
        else:
            data = data.select_dtypes(include=[np.number]).copy()
        
        # Resample to hourly
        data = data.resample('1h').mean()
        data = data.interpolate(method='time', limit=3)
        data = data.dropna()
        
        if len(data) < self.min_training_samples:
            logger.warning(
                f"Insufficient data for {station_id}: "
                f"{len(data)} < {self.min_training_samples} samples"
            )
            # Use simpler model
            model = SimplePersistenceForecaster()
            model.fit(data)
            self.station_models[station_id] = model
        else:
            try:
                model = VARForecaster(VARModelConfig(max_lags=24))
                model.fit(data)
                self.station_models[station_id] = model
            except Exception as e:
                logger.error(f"VAR model training failed for {station_id}: {e}")
                # Fallback to persistence
                model = SimplePersistenceForecaster()
                model.fit(data)
                self.station_models[station_id] = model
        
        self.last_training_times[station_id] = datetime.utcnow()
        logger.info(f"Trained model for station {station_id}")
        
        return True
    
    def generate_forecast(
        self,
        station_id: str,
        horizons: List[int] = None
    ) -> Dict[str, Any]:
        """
        Generate forecasts for a station.
        
        Args:
            station_id: Station identifier
            horizons: List of forecast horizons in hours
        
        Returns:
            Dictionary with forecast data
        """
        if horizons is None:
            horizons = [6, 12, 24, 48, 72]
        
        if station_id not in self.station_models:
            raise ValueError(f"No model available for station {station_id}")
        
        model = self.station_models[station_id]
        max_horizon = max(horizons)
        
        try:
            forecast_df, confidence_intervals = model.predict(steps=max_horizon)
        except Exception as e:
            logger.error(f"Forecast generation failed for {station_id}: {e}")
            raise
        
        # Format results
        forecast_time = datetime.utcnow()
        forecasts = []
        
        for horizon in horizons:
            if horizon <= len(forecast_df):
                point = {
                    "timestamp": (forecast_time + timedelta(hours=horizon)).isoformat(),
                    "horizon_hours": horizon,
                }
                
                # Add variable values
                row = forecast_df.iloc[horizon - 1]
                for col in forecast_df.columns:
                    point[col] = round(float(row[col]), 2)
                    
                    # Add confidence intervals
                    if confidence_intervals and col in confidence_intervals:
                        ci = confidence_intervals[col]
                        if "lower" in ci:
                            point[f"{col}_lower"] = round(float(ci["lower"].iloc[horizon - 1]), 2)
                        if "upper" in ci:
                            point[f"{col}_upper"] = round(float(ci["upper"].iloc[horizon - 1]), 2)
                
                # Confidence score (decreases with horizon)
                point["confidence"] = round(max(0.3, 1.0 - horizon / 120), 2)
                
                forecasts.append(point)
        
        return {
            "station_id": station_id,
            "forecast_time": forecast_time.isoformat(),
            "model_type": type(model).__name__,
            "horizons": horizons,
            "variables": forecast_df.columns.tolist(),
            "forecasts": forecasts
        }
    
    def save_model(self, station_id: str):
        """Save a station model to disk."""
        if station_id not in self.station_models:
            return
        
        model_path = self.model_dir / f"{station_id}_model.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump({
                "model": self.station_models[station_id],
                "training_time": self.last_training_times.get(station_id)
            }, f)
        
        logger.info(f"Saved model for {station_id}")
    
    def load_model(self, station_id: str) -> bool:
        """Load a station model from disk."""
        model_path = self.model_dir / f"{station_id}_model.pkl"
        
        if not model_path.exists():
            return False
        
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            self.station_models[station_id] = data["model"]
            self.last_training_times[station_id] = data.get("training_time")
            
            logger.info(f"Loaded model for {station_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model for {station_id}: {e}")
            return False


class ForecastEvaluator:
    """
    Evaluate forecast accuracy.
    """
    
    @staticmethod
    def compute_metrics(
        actual: pd.Series,
        forecast: pd.Series
    ) -> Dict[str, float]:
        """
        Compute forecast evaluation metrics.
        
        Args:
            actual: Actual values
            forecast: Forecasted values
        
        Returns:
            Dictionary of metrics
        """
        # Align series
        actual = actual.dropna()
        forecast = forecast.reindex(actual.index).dropna()
        
        if len(actual) == 0 or len(forecast) == 0:
            return {}
        
        # Compute metrics
        errors = actual - forecast
        
        mae = np.abs(errors).mean()
        rmse = np.sqrt((errors ** 2).mean())
        mape = (np.abs(errors) / np.abs(actual)).mean() * 100 if (actual != 0).all() else None
        
        # Skill score (compared to persistence)
        persistence_forecast = actual.shift(1)
        persistence_errors = (actual - persistence_forecast).dropna()
        
        if len(persistence_errors) > 0:
            persistence_mse = (persistence_errors ** 2).mean()
            forecast_mse = (errors ** 2).mean()
            skill_score = 1 - forecast_mse / persistence_mse
        else:
            skill_score = None
        
        return {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "mape": round(mape, 2) if mape else None,
            "skill_score": round(skill_score, 3) if skill_score else None,
            "n_samples": len(errors)
        }
    
    @staticmethod
    def evaluate_horizon(
        actual_data: pd.DataFrame,
        forecasts: List[Dict],
        variable: str,
        horizon: int
    ) -> Dict[str, float]:
        """
        Evaluate forecasts at a specific horizon.
        
        Args:
            actual_data: Historical data with actual values
            forecasts: List of forecast dictionaries
            variable: Variable to evaluate
            horizon: Forecast horizon in hours
        """
        # Match forecasts with actuals
        matched_actual = []
        matched_forecast = []
        
        for fc in forecasts:
            if fc.get("horizon_hours") != horizon:
                continue
            
            fc_time = pd.Timestamp(fc["timestamp"])
            
            # Find corresponding actual value
            if fc_time in actual_data.index:
                actual_val = actual_data.loc[fc_time, variable]
                fc_val = fc.get(variable)
                
                if actual_val is not None and fc_val is not None:
                    matched_actual.append(actual_val)
                    matched_forecast.append(fc_val)
        
        if not matched_actual:
            return {}
        
        actual_series = pd.Series(matched_actual)
        forecast_series = pd.Series(matched_forecast)
        
        return ForecastEvaluator.compute_metrics(actual_series, forecast_series)
