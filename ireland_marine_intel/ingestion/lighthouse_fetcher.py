"""
Lighthouse and coastal weather station data fetcher.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

from config.settings import get_settings, LIGHTHOUSE_STATIONS
from .erddap_client import ERDDAPClient

logger = logging.getLogger(__name__)


class LighthouseFetcher:
    """
    Fetcher for coastal lighthouse and Met Éireann station data.
    
    These stations may use different data sources than the buoys.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.stations = LIGHTHOUSE_STATIONS
        self._http_timeout = 30
    
    @property
    def available_stations(self) -> List[str]:
        """Get list of available lighthouse/coastal station IDs."""
        return list(self.stations.keys())
    
    def get_station_metadata(self, station_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific station."""
        return self.stations.get(station_id)
    
    async def fetch_met_eireann_data(
        self,
        station_id: str,
        days_back: int = 7
    ) -> pd.DataFrame:
        """
        Fetch data from Met Éireann ERDDAP server for synoptic stations.
        
        Note: This is a placeholder - actual endpoint may vary based on
        Met Éireann's data availability.
        """
        # Met Éireann data access may require different approach
        # For now, create a structure that can be filled in when data source is confirmed
        
        logger.warning(
            f"Met Éireann ERDDAP fetch not yet implemented for {station_id}. "
            "Using synthetic data for testing."
        )
        
        # Generate synthetic data for testing purposes
        return self._generate_synthetic_lighthouse_data(station_id, days_back)
    
    async def fetch_all_lighthouses(
        self,
        days_back: int = 7
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all available lighthouse/coastal stations.
        
        Returns:
            Dictionary mapping station_id to DataFrame
        """
        results = {}
        
        for station_id in self.stations:
            try:
                df = await self.fetch_met_eireann_data(station_id, days_back)
                if len(df) > 0:
                    # Add metadata
                    metadata = self.stations[station_id]
                    df.attrs["station_id"] = station_id
                    df.attrs["station_name"] = metadata["name"]
                    df.attrs["latitude"] = metadata["lat"]
                    df.attrs["longitude"] = metadata["lon"]
                    df.attrs["station_type"] = metadata["type"]
                    
                    results[station_id] = df
            except Exception as e:
                logger.error(f"Failed to fetch data for {station_id}: {e}")
        
        return results
    
    def _generate_synthetic_lighthouse_data(
        self,
        station_id: str,
        days_back: int
    ) -> pd.DataFrame:
        """
        Generate synthetic lighthouse data for testing.
        
        This creates realistic-looking weather data based on the station's
        location and typical Irish coastal conditions.
        """
        metadata = self.stations.get(station_id)
        if not metadata:
            return pd.DataFrame()
        
        # Generate hourly timestamps
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)
        timestamps = pd.date_range(start=start_time, end=end_time, freq="1h", tz="UTC")
        
        n = len(timestamps)
        
        # Create synthetic data with some realistic patterns
        np.random.seed(hash(station_id) % 2**32)
        
        # Base values with seasonal and diurnal variations
        hour = np.array([t.hour for t in timestamps])
        day = np.array([(t - timestamps[0]).days for t in timestamps])
        
        # Wind: prevailing westerlies with random variations
        wind_base = 15 + 5 * np.sin(2 * np.pi * day / 7)  # Weekly pattern
        wind_speed = wind_base + np.random.normal(0, 5, n)
        wind_speed = np.clip(wind_speed, 0, 80)
        
        # Wind direction: predominantly SW-W with variations
        wind_direction = 240 + np.random.normal(0, 30, n)
        wind_direction = wind_direction % 360
        
        # Temperature: diurnal and seasonal patterns
        temp_base = 10 + 2 * np.sin(2 * np.pi * hour / 24 - np.pi/2)
        air_temperature = temp_base + np.random.normal(0, 1.5, n)
        
        # Pressure: random walk with mean reversion
        pressure = np.zeros(n)
        pressure[0] = 1013
        for i in range(1, n):
            pressure[i] = pressure[i-1] + np.random.normal(0, 1) - 0.01 * (pressure[i-1] - 1013)
        pressure = np.clip(pressure, 970, 1050)
        
        # Humidity: higher when windy and raining
        humidity = 75 + 10 * np.random.randn(n)
        humidity = np.clip(humidity, 40, 100)
        
        # Visibility: depends on weather conditions
        visibility = 20 - 0.1 * wind_speed + np.random.normal(0, 3, n)
        visibility = np.clip(visibility, 0.1, 50)
        
        df = pd.DataFrame({
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "air_temperature": air_temperature,
            "air_pressure": pressure,
            "humidity": humidity,
            "visibility": visibility,
        }, index=timestamps)
        
        df.index.name = "timestamp"
        
        return df


class CoastalDataAggregator:
    """
    Aggregates data from both buoys and lighthouses for regional analysis.
    """
    
    def __init__(self):
        from .buoy_fetcher import BuoyFetcher
        self.buoy_fetcher = BuoyFetcher()
        self.lighthouse_fetcher = LighthouseFetcher()
    
    async def fetch_all_coastal_data(
        self,
        days_back: int = 7
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from all coastal monitoring stations.
        
        Returns combined dictionary of buoy and lighthouse data.
        """
        # Fetch in parallel
        buoy_task = self.buoy_fetcher.fetch_all_buoys(days_back)
        lighthouse_task = self.lighthouse_fetcher.fetch_all_lighthouses(days_back)
        
        buoy_data, lighthouse_data = await asyncio.gather(
            buoy_task, lighthouse_task
        )
        
        # Combine
        all_data = {**buoy_data, **lighthouse_data}
        
        logger.info(f"Fetched data from {len(all_data)} stations")
        
        return all_data
    
    def get_all_station_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all stations."""
        from config.settings import ALL_STATIONS
        return ALL_STATIONS
