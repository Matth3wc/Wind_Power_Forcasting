"""
Data ingestion module for Ireland Marine Weather Intelligence.
"""
from .erddap_client import ERDDAPClient, MarineDataFetcher, fetch_buoy_data_sync
from .buoy_fetcher import BuoyFetcher, BuoyDataProcessor
from .lighthouse_fetcher import LighthouseFetcher, CoastalDataAggregator
from .bathymetry_fetcher import BathymetryFetcher, BathymetryAnalyzer
from .scheduler import DataScheduler, RealTimeDataManager

__all__ = [
    "ERDDAPClient",
    "MarineDataFetcher",
    "fetch_buoy_data_sync",
    "BuoyFetcher",
    "BuoyDataProcessor",
    "LighthouseFetcher",
    "CoastalDataAggregator",
    "BathymetryFetcher",
    "BathymetryAnalyzer",
    "DataScheduler",
    "RealTimeDataManager",
]
