"""
Data collection scheduler for periodic data fetching.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from config.settings import get_settings
from .buoy_fetcher import BuoyFetcher
from .lighthouse_fetcher import LighthouseFetcher, CoastalDataAggregator
from .bathymetry_fetcher import BathymetryFetcher

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Result of a data fetch operation."""
    success: bool
    timestamp: datetime
    source: str
    records_fetched: int = 0
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


class DataScheduler:
    """
    Scheduler for periodic data collection from all sources.
    
    Handles:
    - Regular data fetching at configured intervals
    - Error handling and retry logic
    - Status tracking and reporting
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.scheduler = AsyncIOScheduler()
        
        # Fetchers
        self.buoy_fetcher = BuoyFetcher()
        self.lighthouse_fetcher = LighthouseFetcher()
        self.aggregator = CoastalDataAggregator()
        self.bathymetry_fetcher = BathymetryFetcher()
        
        # Status tracking
        self._fetch_history: List[FetchResult] = []
        self._last_data: Dict[str, Any] = {}
        self._callbacks: List[Callable] = []
        
        # Running state
        self._is_running = False
    
    def add_callback(self, callback: Callable):
        """Add a callback to be called when new data is fetched."""
        self._callbacks.append(callback)
    
    async def _notify_callbacks(self, data: Dict[str, Any]):
        """Notify all registered callbacks of new data."""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    async def fetch_all_data(self) -> Dict[str, Any]:
        """
        Fetch data from all sources.
        
        Returns:
            Dictionary with data from all sources
        """
        start_time = datetime.utcnow()
        results = {}
        
        try:
            # Fetch buoy data
            logger.info("Fetching buoy data...")
            buoy_data = await self.buoy_fetcher.fetch_all_buoys(days_back=1)
            results["buoys"] = buoy_data
            
            self._fetch_history.append(FetchResult(
                success=True,
                timestamp=datetime.utcnow(),
                source="buoys",
                records_fetched=sum(len(df) for df in buoy_data.values()),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds()
            ))
            
        except Exception as e:
            logger.error(f"Error fetching buoy data: {e}")
            self._fetch_history.append(FetchResult(
                success=False,
                timestamp=datetime.utcnow(),
                source="buoys",
                error_message=str(e),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds()
            ))
        
        try:
            # Fetch lighthouse data
            logger.info("Fetching lighthouse data...")
            lighthouse_data = await self.lighthouse_fetcher.fetch_all_lighthouses(days_back=1)
            results["lighthouses"] = lighthouse_data
            
        except Exception as e:
            logger.error(f"Error fetching lighthouse data: {e}")
        
        # Cache the results
        self._last_data = results
        self._last_data["timestamp"] = datetime.utcnow()
        
        # Notify callbacks
        await self._notify_callbacks(results)
        
        return results
    
    async def fetch_buoy_data_job(self):
        """Scheduled job for fetching buoy data."""
        logger.info("Running scheduled buoy data fetch")
        
        try:
            data = await self.buoy_fetcher.fetch_all_buoys(days_back=1)
            
            self._last_data["buoys"] = data
            self._last_data["buoys_timestamp"] = datetime.utcnow()
            
            record_count = sum(len(df) for df in data.values())
            logger.info(f"Fetched {record_count} records from {len(data)} buoys")
            
            # Notify callbacks
            await self._notify_callbacks({"buoys": data})
            
        except Exception as e:
            logger.error(f"Scheduled buoy fetch failed: {e}")
    
    async def fetch_lighthouse_data_job(self):
        """Scheduled job for fetching lighthouse data."""
        logger.info("Running scheduled lighthouse data fetch")
        
        try:
            data = await self.lighthouse_fetcher.fetch_all_lighthouses(days_back=1)
            
            self._last_data["lighthouses"] = data
            self._last_data["lighthouses_timestamp"] = datetime.utcnow()
            
            logger.info(f"Fetched data from {len(data)} lighthouses")
            
            # Notify callbacks
            await self._notify_callbacks({"lighthouses": data})
            
        except Exception as e:
            logger.error(f"Scheduled lighthouse fetch failed: {e}")
    
    def start(self):
        """Start the scheduler with configured jobs."""
        if self._is_running:
            logger.warning("Scheduler is already running")
            return
        
        fetch_interval = self.settings.fetch_interval_minutes
        
        # Add buoy fetch job (every 15 minutes by default)
        self.scheduler.add_job(
            self.fetch_buoy_data_job,
            IntervalTrigger(minutes=fetch_interval),
            id="buoy_fetch",
            name="Buoy Data Fetch",
            replace_existing=True
        )
        
        # Add lighthouse fetch job (every 30 minutes)
        self.scheduler.add_job(
            self.fetch_lighthouse_data_job,
            IntervalTrigger(minutes=fetch_interval * 2),
            id="lighthouse_fetch",
            name="Lighthouse Data Fetch",
            replace_existing=True
        )
        
        # Start the scheduler
        self.scheduler.start()
        self._is_running = True
        
        logger.info(f"Scheduler started with {fetch_interval} minute intervals")
    
    def stop(self):
        """Stop the scheduler."""
        if self._is_running:
            self.scheduler.shutdown()
            self._is_running = False
            logger.info("Scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            "is_running": self._is_running,
            "jobs": [
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                }
                for job in self.scheduler.get_jobs()
            ] if self._is_running else [],
            "last_fetch": {
                "buoys": self._last_data.get("buoys_timestamp"),
                "lighthouses": self._last_data.get("lighthouses_timestamp"),
            },
            "recent_history": [
                {
                    "source": r.source,
                    "success": r.success,
                    "timestamp": r.timestamp.isoformat(),
                    "records": r.records_fetched,
                    "error": r.error_message,
                }
                for r in self._fetch_history[-10:]
            ]
        }
    
    def get_latest_data(self) -> Dict[str, Any]:
        """Get the most recently fetched data."""
        return self._last_data


class RealTimeDataManager:
    """
    Manager for real-time data distribution.
    
    Handles:
    - Caching latest readings
    - Broadcasting updates to connected clients
    - Data aggregation and summarization
    """
    
    def __init__(self):
        self.scheduler = DataScheduler()
        self._latest_readings: Dict[str, Any] = {}
        self._subscribers: List[asyncio.Queue] = []
        
        # Register callback
        self.scheduler.add_callback(self._on_data_update)
    
    async def _on_data_update(self, data: Dict[str, Any]):
        """Called when new data is fetched."""
        # Update latest readings
        self._update_latest_readings(data)
        
        # Notify subscribers
        update_message = {
            "type": "data_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": self._latest_readings
        }
        
        for queue in self._subscribers:
            try:
                queue.put_nowait(update_message)
            except asyncio.QueueFull:
                logger.warning("Subscriber queue full, dropping message")
    
    def _update_latest_readings(self, data: Dict[str, Any]):
        """Update the latest readings cache."""
        for source, station_data in data.items():
            if isinstance(station_data, dict):
                for station_id, df in station_data.items():
                    if hasattr(df, 'iloc') and len(df) > 0:
                        latest = df.iloc[-1].to_dict()
                        latest["timestamp"] = df.index[-1].isoformat() if hasattr(df.index[-1], 'isoformat') else str(df.index[-1])
                        self._latest_readings[station_id] = latest
    
    def subscribe(self) -> asyncio.Queue:
        """Subscribe to real-time updates."""
        queue = asyncio.Queue(maxsize=100)
        self._subscribers.append(queue)
        return queue
    
    def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from updates."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)
    
    def get_latest_readings(self) -> Dict[str, Any]:
        """Get the latest readings for all stations."""
        return self._latest_readings.copy()
    
    def start(self):
        """Start the data manager."""
        self.scheduler.start()
    
    def stop(self):
        """Stop the data manager."""
        self.scheduler.stop()
