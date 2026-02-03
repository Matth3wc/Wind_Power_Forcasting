"""
Buoy data fetcher with station-specific handling.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

from config.settings import get_settings, BUOY_STATIONS, COASTAL_STATIONS
from .erddap_client import ERDDAPClient, MarineDataFetcher

logger = logging.getLogger(__name__)


class BuoyFetcher:
    """
    Specialized fetcher for Irish weather buoy data.
    
    Handles data quality checks, gap filling, and station-specific processing.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.stations = {**BUOY_STATIONS, **COASTAL_STATIONS}
        self._fetcher = MarineDataFetcher()
    
    @property
    def available_stations(self) -> List[str]:
        """Get list of available buoy station IDs."""
        return list(self.stations.keys())
    
    def get_station_metadata(self, station_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific station."""
        return self.stations.get(station_id)
    
    async def fetch_single_buoy(
        self,
        station_id: str,
        days_back: int = 7,
        apply_qc: bool = True
    ) -> pd.DataFrame:
        """
        Fetch data for a single buoy with quality control.
        
        Args:
            station_id: Buoy station identifier
            days_back: Number of days of historical data
            apply_qc: Whether to apply quality control checks
        
        Returns:
            Cleaned DataFrame with buoy data
        """
        if station_id not in self.stations:
            raise ValueError(f"Unknown station: {station_id}")
        
        df = await self._fetcher.get_historical_data(station_id, days_back)
        
        if apply_qc and len(df) > 0:
            df = self._apply_quality_control(df)
        
        # Add station metadata
        metadata = self.stations[station_id]
        df.attrs["station_id"] = station_id
        df.attrs["station_name"] = metadata["name"]
        df.attrs["latitude"] = metadata["lat"]
        df.attrs["longitude"] = metadata["lon"]
        df.attrs["station_type"] = metadata["type"]
        
        return df
    
    async def fetch_all_buoys(
        self,
        days_back: int = 7,
        apply_qc: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all available buoys.
        
        Args:
            days_back: Number of days of historical data
            apply_qc: Whether to apply quality control
        
        Returns:
            Dictionary mapping station_id to DataFrame
        """
        async with ERDDAPClient() as client:
            data = await client.fetch_all_buoys(
                list(self.stations.keys()),
                days_back
            )
        
        results = {}
        for station_id, df in data.items():
            if df is not None and len(df) > 0:
                if apply_qc:
                    df = self._apply_quality_control(df)
                
                # Add metadata
                metadata = self.stations[station_id]
                df.attrs["station_id"] = station_id
                df.attrs["station_name"] = metadata["name"]
                df.attrs["latitude"] = metadata["lat"]
                df.attrs["longitude"] = metadata["lon"]
                
                results[station_id] = df
        
        return results
    
    async def fetch_latest_readings(self) -> pd.DataFrame:
        """
        Get the most recent reading from each buoy.
        
        Returns:
            DataFrame with one row per station
        """
        return await self._fetcher.get_latest_readings(
            list(self.stations.keys())
        )
    
    def _apply_quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply quality control checks to buoy data.
        
        - Removes physically impossible values
        - Flags suspicious spikes
        - Interpolates small gaps
        """
        from config.settings import WEATHER_VARIABLES
        
        df = df.copy()
        
        # Define valid ranges for each variable
        valid_ranges = {
            "wind_speed": (0, 150),
            "wind_direction": (0, 360),
            "wave_height": (0, 30),
            "wave_period": (0, 30),
            "air_temperature": (-30, 50),
            "air_pressure": (900, 1100),
            "sea_temperature": (-2, 35),
        }
        
        # Remove out-of-range values
        for col, (min_val, max_val) in valid_ranges.items():
            if col in df.columns:
                mask = (df[col] < min_val) | (df[col] > max_val)
                if mask.any():
                    logger.debug(f"Removing {mask.sum()} out-of-range values in {col}")
                    df.loc[mask, col] = np.nan
        
        # Remove spikes (values > 3 std from rolling mean)
        for col in df.select_dtypes(include=[np.number]).columns:
            if len(df) > 10:
                rolling_mean = df[col].rolling(window=5, center=True, min_periods=2).mean()
                rolling_std = df[col].rolling(window=5, center=True, min_periods=2).std()
                
                # Flag values more than 3 std from rolling mean
                spike_mask = np.abs(df[col] - rolling_mean) > 3 * rolling_std
                if spike_mask.any():
                    logger.debug(f"Flagging {spike_mask.sum()} spikes in {col}")
                    df.loc[spike_mask, col] = np.nan
        
        # Interpolate small gaps (up to 3 consecutive missing values)
        df = df.infer_objects(copy=False)
        df = df.interpolate(method="time", limit=3)
        
        return df
    
    def compute_derived_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived variables from raw measurements.
        
        Adds:
            - wind_u, wind_v: Wind vector components
            - wave_power: Estimated wave power
            - beaufort: Beaufort scale classification
        """
        df = df.copy()
        
        # Wind vector components
        if "wind_speed" in df.columns and "wind_direction" in df.columns:
            wind_dir_rad = np.deg2rad(df["wind_direction"])
            df["wind_u"] = -df["wind_speed"] * np.sin(wind_dir_rad)
            df["wind_v"] = -df["wind_speed"] * np.cos(wind_dir_rad)
        
        # Wave power (simplified formula: P ≈ 0.5 * H² * T in kW/m)
        if "wave_height" in df.columns and "wave_period" in df.columns:
            df["wave_power"] = 0.5 * df["wave_height"] ** 2 * df["wave_period"]
        
        # Beaufort scale
        if "wind_speed" in df.columns:
            # Convert knots to m/s for Beaufort scale
            wind_ms = df["wind_speed"] * 0.514444
            
            beaufort_thresholds = [0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]
            
            def to_beaufort(speed):
                if pd.isna(speed):
                    return np.nan
                for i, threshold in enumerate(beaufort_thresholds):
                    if speed < threshold:
                        return i
                return 12
            
            df["beaufort"] = wind_ms.apply(to_beaufort)
        
        return df


class BuoyDataProcessor:
    """
    Process and transform buoy data for analysis and visualization.
    """
    
    @staticmethod
    def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
        """Resample data to hourly frequency."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].resample("1h").mean()
    
    @staticmethod
    def compute_statistics(df: pd.DataFrame, window: str = "24H") -> pd.DataFrame:
        """
        Compute rolling statistics.
        
        Args:
            df: Input DataFrame
            window: Rolling window size (e.g., '24H', '7D')
        
        Returns:
            DataFrame with rolling mean, std, min, max for each variable
        """
        result = pd.DataFrame(index=df.index)
        
        for col in df.select_dtypes(include=[np.number]).columns:
            result[f"{col}_mean"] = df[col].rolling(window).mean()
            result[f"{col}_std"] = df[col].rolling(window).std()
            result[f"{col}_min"] = df[col].rolling(window).min()
            result[f"{col}_max"] = df[col].rolling(window).max()
        
        return result
    
    @staticmethod
    def align_stations(
        data: Dict[str, pd.DataFrame],
        resample_freq: str = "1h"
    ) -> pd.DataFrame:
        """
        Align data from multiple stations to common timestamps.
        
        Args:
            data: Dictionary of station_id -> DataFrame
            resample_freq: Resample frequency
        
        Returns:
            DataFrame with MultiIndex columns (station_id, variable)
        """
        resampled = {}
        
        for station_id, df in data.items():
            if len(df) > 0:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                resampled_df = df[numeric_cols].resample(resample_freq).mean()
                resampled[station_id] = resampled_df
        
        if not resampled:
            return pd.DataFrame()
        
        combined = pd.concat(resampled, axis=1)
        
        # Find the common time range where most stations have data
        valid_counts = combined.notna().sum(axis=1)
        threshold = len(resampled) * 0.5  # At least 50% of stations
        combined = combined[valid_counts >= threshold]
        
        return combined
