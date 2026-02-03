"""
ERDDAP Client for fetching marine and meteorological data.
Provides a unified interface to multiple ERDDAP servers.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import pandas as pd
import numpy as np
from erddapy import ERDDAP
import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ERDDAPDataset:
    """Represents an ERDDAP dataset configuration."""
    server_url: str
    dataset_id: str
    variables: List[str]
    constraints: Dict[str, Any]
    protocol: str = "tabledap"


class ERDDAPClient:
    """
    Async ERDDAP client for fetching oceanographic and meteorological data.
    
    Supports multiple ERDDAP servers and provides caching and rate limiting.
    """
    
    def __init__(self, timeout: int = 30):
        self.settings = get_settings()
        self.timeout = timeout
        self._cache: Dict[str, tuple] = {}  # (data, timestamp)
        self._http_client: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._http_client:
            await self._http_client.aclose()
    
    def _create_erddap_instance(self, server_url: str, dataset_id: str) -> ERDDAP:
        """Create an ERDDAP instance for a specific server and dataset."""
        e = ERDDAP(
            server=server_url,
            protocol="tabledap",
        )
        e.dataset_id = dataset_id
        return e
    
    def _build_erddap_url(
        self,
        server_url: str,
        dataset_id: str,
        variables: List[str],
        constraints: Dict[str, Any],
        response_format: str = "csv"
    ) -> str:
        """Build an ERDDAP data request URL."""
        base_url = f"{server_url}/tabledap/{dataset_id}.{response_format}"
        
        # Build variable list
        var_str = ",".join(variables)
        
        # Build constraints - encode operators and handle quoting
        # ERDDAP requires > and < to be URL-encoded in the query string
        constraint_parts = []
        for key, value in constraints.items():
            # URL-encode the operators in the key (>= becomes %3E=, <= becomes %3C=)
            encoded_key = key.replace(">", "%3E").replace("<", "%3C")
            
            # Time constraints should not have quotes around the value
            if 'time' in key.lower():
                constraint_parts.append(f'{encoded_key}{value}')
            elif isinstance(value, str):
                constraint_parts.append(f'{encoded_key}"{value}"')
            else:
                constraint_parts.append(f"{encoded_key}{value}")
        
        constraint_str = "&".join(constraint_parts)
        
        return f"{base_url}?{var_str}&{constraint_str}"
    
    async def fetch_csv_async(self, url: str) -> pd.DataFrame:
        """Fetch CSV data from URL asynchronously."""
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        
        try:
            response = await self._http_client.get(url)
            response.raise_for_status()
            
            # Parse CSV from response
            from io import StringIO
            df = pd.read_csv(StringIO(response.text), skiprows=[1])  # Skip units row
            return df
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            raise
    
    def fetch_sync(
        self,
        server_url: str,
        dataset_id: str,
        variables: List[str],
        constraints: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Synchronous fetch using erddapy library.
        
        Args:
            server_url: ERDDAP server URL
            dataset_id: Dataset identifier
            variables: List of variable names to fetch
            constraints: Dictionary of constraints (e.g., time ranges, station IDs)
        
        Returns:
            DataFrame with fetched data
        """
        e = self._create_erddap_instance(server_url, dataset_id)
        e.variables = variables
        e.constraints = constraints
        
        try:
            df = e.to_pandas(
                index_col="time (UTC)" if "time" in variables else None,
                parse_dates=True
            )
            return df.dropna(how="all")
        except Exception as ex:
            logger.error(f"Error fetching from {dataset_id}: {ex}")
            raise
    
    async def fetch_marine_ie_buoy_data(
        self,
        station_id: str,
        days_back: int = 7
    ) -> pd.DataFrame:
        """
        Fetch data from Marine Institute Irish Weather Buoy Network.
        
        Args:
            station_id: Station identifier (e.g., 'M5', 'IL1')
            days_back: Number of days of historical data to fetch
        
        Returns:
            DataFrame with buoy measurements
        """
        variables = [
            "time", "station_id", "latitude", "longitude",
            "WindSpeed", "WindDirection", "Gust",
            "AirTemperature", "AtmosphericPressure",
            "WaveHeight", "WavePeriod", "Hmax", "Tp", "MeanWaveDirection",
            "SeaTemperature", "RelativeHumidity"
        ]
        
        # Calculate time constraint - use ISO format for better compatibility
        start_time = datetime.utcnow() - timedelta(days=days_back)
        time_constraint = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        constraints = {
            "time>=": time_constraint,
            "station_id=": station_id,
        }
        
        url = self._build_erddap_url(
            self.settings.marine_ie_erddap_url,
            "IWBNetwork",
            variables,
            constraints
        )
        
        logger.info(f"Fetching buoy data for {station_id} from Marine.ie")
        
        df = await self.fetch_csv_async(url)
        
        # Standardize column names
        df = self._standardize_column_names(df)
        
        return df
    
    async def fetch_all_buoys(
        self,
        station_ids: List[str],
        days_back: int = 7
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple buoys concurrently.
        
        Args:
            station_ids: List of station IDs
            days_back: Number of days of historical data
        
        Returns:
            Dictionary mapping station_id to DataFrame
        """
        results = {}
        
        # Use semaphore to limit concurrent requests
        sem = asyncio.Semaphore(self.settings.max_concurrent_requests)
        
        async def fetch_with_semaphore(station_id: str):
            async with sem:
                try:
                    df = await self.fetch_marine_ie_buoy_data(station_id, days_back)
                    return station_id, df
                except Exception as e:
                    logger.warning(f"Failed to fetch {station_id}: {e}")
                    return station_id, None
        
        tasks = [fetch_with_semaphore(sid) for sid in station_ids]
        completed = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in completed:
            if isinstance(result, tuple) and result[1] is not None:
                station_id, df = result
                results[station_id] = df
        
        return results
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to a consistent format."""
        rename_map = {
            "time (UTC)": "timestamp",
            "time": "timestamp",
            "station_id": "station_id",
            "latitude (degrees_north)": "latitude",
            "latitude": "latitude",
            "longitude (degrees_east)": "longitude",
            "longitude": "longitude",
            "WindSpeed (knots)": "wind_speed",
            "WindSpeed": "wind_speed",
            "WindDirection (degrees_true)": "wind_direction",
            "WindDirection (degrees true)": "wind_direction",
            "WindDirection": "wind_direction",
            "Gust (knots)": "wind_gust",
            "Gust": "wind_gust",
            "WindGust (knots)": "wind_gust",
            "WindGust": "wind_gust",
            "AirTemperature (degrees_C)": "air_temperature",
            "AirTemperature": "air_temperature",
            "AtmosphericPressure (millibars)": "air_pressure",
            "AtmosphericPressure": "air_pressure",
            "WaveHeight (meters)": "wave_height",
            "WaveHeight": "wave_height",
            "WavePeriod (seconds)": "wave_period_mean",
            "WavePeriod": "wave_period_mean",
            "Hmax (meters)": "wave_height_max",
            "Hmax": "wave_height_max",
            "Tp (seconds)": "wave_period",
            "Tp": "wave_period",
            "MeanWaveDirection (degrees_true)": "wave_direction",
            "MeanWaveDirection": "wave_direction",
            "SeaTemperature (degrees_C)": "sea_temperature",
            "SeaTemperature": "sea_temperature",
            "RelativeHumidity (percent)": "relative_humidity",
            "RelativeHumidity": "relative_humidity",
        }
        
        # Only rename columns that exist
        existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=existing_renames)
        
        # Parse timestamp if it's a column (not index)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        
        return df
    
    def get_available_datasets(self, server_url: str) -> List[Dict[str, str]]:
        """List available datasets on an ERDDAP server."""
        e = ERDDAP(server=server_url, protocol="tabledap")
        
        try:
            # This returns metadata about available datasets
            search_url = f"{server_url}/search/advanced.json?searchFor=&page=1&itemsPerPage=1000"
            import requests
            response = requests.get(search_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            datasets = []
            if "table" in data:
                rows = data["table"]["rows"]
                column_names = data["table"]["columnNames"]
                
                for row in rows:
                    dataset_info = dict(zip(column_names, row))
                    datasets.append({
                        "id": dataset_info.get("Dataset ID", ""),
                        "title": dataset_info.get("Title", ""),
                        "institution": dataset_info.get("Institution", ""),
                    })
            
            return datasets
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            return []


class MarineDataFetcher:
    """
    High-level interface for fetching and processing marine data.
    
    Combines data from multiple sources and provides clean, unified output.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = ERDDAPClient(timeout=self.settings.request_timeout_seconds)
    
    async def get_latest_readings(
        self,
        station_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get the most recent reading from each station.
        
        Args:
            station_ids: Optional list of specific stations. If None, fetches all.
        
        Returns:
            DataFrame with one row per station containing latest values
        """
        from config.settings import ALL_STATIONS
        
        if station_ids is None:
            # Only fetch buoys (they have ERDDAP data)
            station_ids = [s for s in ALL_STATIONS.keys() 
                         if ALL_STATIONS[s]["type"] in ["offshore_buoy", "coastal_buoy"]]
        
        async with self.client:
            data = await self.client.fetch_all_buoys(station_ids, days_back=1)
        
        # Extract latest reading from each
        latest_readings = []
        for station_id, df in data.items():
            if df is not None and len(df) > 0:
                latest = df.iloc[-1].copy()
                latest["station_id"] = station_id
                latest["timestamp"] = df.index[-1]
                latest_readings.append(latest)
        
        if not latest_readings:
            return pd.DataFrame()
        
        result = pd.DataFrame(latest_readings)
        result = result.set_index("station_id")
        
        return result
    
    async def get_historical_data(
        self,
        station_id: str,
        days_back: int = 30,
        resample_freq: Optional[str] = "1H"
    ) -> pd.DataFrame:
        """
        Get historical data for a station, optionally resampled.
        
        Args:
            station_id: Station identifier
            days_back: Number of days of history
            resample_freq: Pandas resample frequency (e.g., '1H', '30T')
        
        Returns:
            DataFrame with historical measurements
        """
        async with self.client:
            df = await self.client.fetch_marine_ie_buoy_data(station_id, days_back)
        
        if resample_freq and len(df) > 0:
            # Resample to regular intervals
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_cols].resample(resample_freq).mean()
            df = df.interpolate(method="time", limit=3)
        
        return df
    
    async def get_regional_data(
        self,
        station_ids: List[str],
        days_back: int = 7,
        resample_freq: str = "1H"
    ) -> pd.DataFrame:
        """
        Get data for multiple stations with aligned timestamps.
        
        Args:
            station_ids: List of station IDs
            days_back: Number of days of history
            resample_freq: Resample frequency
        
        Returns:
            DataFrame with MultiIndex columns (station_id, variable)
        """
        async with self.client:
            data = await self.client.fetch_all_buoys(station_ids, days_back)
        
        # Resample each to regular intervals
        resampled = {}
        for station_id, df in data.items():
            if df is not None and len(df) > 0:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                resampled_df = df[numeric_cols].resample(resample_freq).mean()
                resampled_df = resampled_df.interpolate(method="time", limit=3)
                resampled[station_id] = resampled_df
        
        if not resampled:
            return pd.DataFrame()
        
        # Combine into multi-index DataFrame
        combined = pd.concat(resampled, axis=1)
        
        # Find common time range
        combined = combined.dropna(how="all")
        
        return combined


# Convenience function for sync usage
def fetch_buoy_data_sync(station_id: str, days_back: int = 7) -> pd.DataFrame:
    """
    Synchronous convenience function for fetching buoy data.
    
    Args:
        station_id: Station identifier
        days_back: Number of days of historical data
    
    Returns:
        DataFrame with buoy measurements
    """
    settings = get_settings()
    client = ERDDAPClient()
    
    return client.fetch_sync(
        settings.marine_ie_erddap_url,
        "IWBNetwork",
        ["time", "station_id", "WindSpeed", "WindDirection", "WindGust",
         "AirTemperature", "AtmosphericPressure", "WaveHeight", "Hmax", "Tp",
         "MeanWaveDirection", "SeaTemperature"],
        {"time>=": f"now-{days_back}days", "station_id=": station_id}
    )
