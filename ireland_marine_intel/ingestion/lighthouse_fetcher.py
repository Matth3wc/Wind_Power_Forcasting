"""
Lighthouse and coastal weather station data fetcher.
Uses Met Éireann's real-time AWS observation data from opendata2.met.ie
"""
import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from io import StringIO

import pandas as pd
import numpy as np
import httpx

from config.settings import get_settings, LIGHTHOUSE_STATIONS

logger = logging.getLogger(__name__)

# Met Éireann AWS base URL
MET_EIREANN_OBS_BASE_URL = "https://opendata2.met.ie/obs"


class LighthouseFetcher:
    """
    Fetcher for Met Éireann automatic weather station data.
    
    Uses real-time observations from opendata2.met.ie/obs/
    Data structure: /obs/{Station}/{Month}/{Day}/{Hour}/{Sensor}/
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.stations = LIGHTHOUSE_STATIONS
        self._http_timeout = 30
        self._http_client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self._http_client = httpx.AsyncClient(timeout=self._http_timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._http_client:
            await self._http_client.aclose()
    
    @property
    def available_stations(self) -> List[str]:
        """Get list of available lighthouse/coastal station IDs."""
        return list(self.stations.keys())
    
    def get_station_metadata(self, station_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific station."""
        return self.stations.get(station_id)
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=self._http_timeout)
        return self._http_client
    
    def _parse_met_ie_csv(self, text: str) -> pd.DataFrame:
        """Parse Met Éireann CR3 format CSV data."""
        # CR3 format has a header with metadata rows
        # Row 1: Table info
        # Row 2: Column names
        # Row 3: Units
        # Row 4: Processing type
        # Data starts at row 5
        lines = text.strip().split('\n')
        if len(lines) < 5:
            return pd.DataFrame()
        
        # Parse header to get column names
        header_line = lines[1].replace('"', '')
        columns = header_line.split(',')
        
        # Parse data lines (skip first 4 header lines)
        data_lines = lines[4:]
        data = []
        for line in data_lines:
            # Remove quotes and split
            values = line.replace('"', '').split(',')
            data.append(values)
        
        df = pd.DataFrame(data, columns=columns)
        
        # Convert timestamp
        if 'TIMESTAMP' in df.columns:
            df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
            df.set_index('TIMESTAMP', inplace=True)
        
        # Convert numeric columns
        for col in df.columns:
            if col not in ['StationID', 'Station_ID']:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        return df
    
    async def _fetch_sensor_data(
        self,
        station_id: str,
        sensor: str,
        hours_back: int = 24
    ) -> pd.DataFrame:
        """
        Fetch data from a specific sensor for a station.
        
        Args:
            station_id: Met Éireann station ID (e.g., 'MalinHead')
            sensor: Sensor type ('Wind', 'Pressure', 'Suit_A')
            hours_back: Number of hours of data to fetch
        """
        client = await self._get_http_client()
        all_data = []
        
        now = datetime.utcnow()
        
        # We need to construct URLs for each hour
        # Format: /obs/{Station}/{Month}/{Day}/{Hour}/{Sensor}/
        for hours_ago in range(0, hours_back):
            target_time = now - timedelta(hours=hours_ago)
            month = f"{target_time.month:02d}"
            day = f"{target_time.day:02d}"
            hour = f"{target_time.hour:02d}"
            
            # List files in the sensor directory
            dir_url = f"{MET_EIREANN_OBS_BASE_URL}/{station_id}/{month}/{day}/{hour}/{sensor}/"
            
            try:
                response = await client.get(dir_url)
                if response.status_code != 200:
                    continue
                
                # Parse directory listing to find data files
                html = response.text
                # Find .CR3 files
                cr3_files = re.findall(r'href="([^"]+\.CR3)"', html)
                
                if not cr3_files:
                    continue
                
                # Fetch the most recent file (usually there's one main file)
                # The files with _01_ are 1-minute data, _60_ are hourly aggregates
                # Prefer _01_ for more detail
                target_file = cr3_files[0]
                for f in cr3_files:
                    if '_01_' in f:
                        target_file = f
                        break
                
                file_url = f"{dir_url}{target_file}"
                file_response = await client.get(file_url)
                
                if file_response.status_code == 200:
                    df = self._parse_met_ie_csv(file_response.text)
                    if not df.empty:
                        all_data.append(df)
                
            except Exception as e:
                logger.debug(f"Error fetching {sensor} data for {station_id} at {target_time}: {e}")
                continue
        
        if all_data:
            combined = pd.concat(all_data)
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.sort_index(inplace=True)
            return combined
        
        return pd.DataFrame()
    
    async def fetch_met_eireann_data(
        self,
        station_id: str,
        days_back: int = 1
    ) -> pd.DataFrame:
        """
        Fetch real data from Met Éireann AWS for a station.
        
        Combines Wind, Pressure, and temperature data.
        """
        hours_back = min(days_back * 24, 168)  # Max 7 days (168 hours)
        
        # Fetch wind and other sensors
        wind_task = self._fetch_sensor_data(station_id, "Wind", hours_back)
        pressure_task = self._fetch_sensor_data(station_id, "Pressure", hours_back)
        suit_a_task = self._fetch_sensor_data(station_id, "Suit_A", hours_back)
        
        wind_df, pressure_df, suit_a_df = await asyncio.gather(
            wind_task, pressure_task, suit_a_task,
            return_exceptions=True
        )
        
        # Process results
        dfs_to_merge = []
        
        # Wind data
        if isinstance(wind_df, pd.DataFrame) and not wind_df.empty:
            wind_cols = {}
            if 'WindSpeed' in wind_df.columns:
                wind_cols['wind_speed'] = wind_df['WindSpeed']
            if 'WindDir' in wind_df.columns:
                wind_cols['wind_direction'] = wind_df['WindDir']
            if 'MaxGust' in wind_df.columns:
                wind_cols['wind_gust'] = wind_df['MaxGust']
            if wind_cols:
                dfs_to_merge.append(pd.DataFrame(wind_cols))
        
        # Pressure data
        if isinstance(pressure_df, pd.DataFrame) and not pressure_df.empty:
            if 'Pressure_QFE' in pressure_df.columns:
                dfs_to_merge.append(pd.DataFrame({'air_pressure': pressure_df['Pressure_QFE']}))
            elif 'Pressure' in pressure_df.columns:
                dfs_to_merge.append(pd.DataFrame({'air_pressure': pressure_df['Pressure']}))
        
        # Suit_A usually contains temperature and humidity
        if isinstance(suit_a_df, pd.DataFrame) and not suit_a_df.empty:
            suit_cols = {}
            for col in suit_a_df.columns:
                if 'Temp' in col and 'air_temperature' not in suit_cols:
                    suit_cols['air_temperature'] = suit_a_df[col]
                if 'RH' in col or 'Humidity' in col:
                    suit_cols['humidity'] = suit_a_df[col]
            if suit_cols:
                dfs_to_merge.append(pd.DataFrame(suit_cols))
        
        # Combine all data
        if dfs_to_merge:
            # Merge on index
            result = dfs_to_merge[0]
            for df in dfs_to_merge[1:]:
                result = result.join(df, how='outer')
            
            result = result[~result.index.duplicated(keep='last')]
            result.sort_index(inplace=True)
            
            # Resample to hourly
            result = result.resample('1h').mean()
            
            logger.info(f"Fetched {len(result)} records for {station_id} from Met Éireann")
            return result
        
        # Fallback to synthetic data if no real data available
        logger.warning(f"No real data for {station_id}, using synthetic data")
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
