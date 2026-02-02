"""
Weather flow analysis and pattern detection.
"""
import logging
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import correlate, find_peaks
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)


@dataclass
class FlowVector:
    """Represents a weather flow vector at a location."""
    latitude: float
    longitude: float
    u_component: float  # Eastward component
    v_component: float  # Northward component
    magnitude: float
    direction: float  # Meteorological direction (from)
    timestamp: datetime


class WeatherFlowField:
    """
    Represents a 2D weather flow field over Ireland.
    
    Can be used for visualization and interpolation.
    """
    
    def __init__(
        self,
        lat_bounds: Tuple[float, float] = (51.0, 56.0),
        lon_bounds: Tuple[float, float] = (-11.0, -5.0),
        resolution: float = 0.25
    ):
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.resolution = resolution
        
        # Create grid
        self.lats = np.arange(lat_bounds[0], lat_bounds[1], resolution)
        self.lons = np.arange(lon_bounds[0], lon_bounds[1], resolution)
        self.grid_lats, self.grid_lons = np.meshgrid(self.lats, self.lons)
        
        self.u_field: Optional[np.ndarray] = None
        self.v_field: Optional[np.ndarray] = None
    
    def interpolate_from_stations(
        self,
        stations: List[Dict[str, Any]],
        method: str = "linear"
    ):
        """
        Interpolate wind field from station observations.
        
        Args:
            stations: List of station dicts with lat, lon, wind_speed, wind_direction
            method: Interpolation method ('linear', 'cubic', 'nearest')
        """
        if not stations:
            return
        
        # Extract station data
        points = []
        u_values = []
        v_values = []
        
        for st in stations:
            if all(k in st for k in ['lat', 'lon', 'wind_speed', 'wind_direction']):
                points.append([st['lon'], st['lat']])
                
                # Convert to U/V components
                speed = st['wind_speed']
                direction = st['wind_direction']
                
                # Meteorological convention: direction is where wind comes FROM
                dir_rad = np.radians(direction)
                u = -speed * np.sin(dir_rad)
                v = -speed * np.cos(dir_rad)
                
                u_values.append(u)
                v_values.append(v)
        
        if len(points) < 3:
            logger.warning("Not enough stations for interpolation")
            return
        
        points = np.array(points)
        
        # Interpolate to grid
        grid_points = np.column_stack([self.grid_lons.ravel(), self.grid_lats.ravel()])
        
        try:
            self.u_field = griddata(
                points, u_values, grid_points, method=method
            ).reshape(self.grid_lons.shape)
            
            self.v_field = griddata(
                points, v_values, grid_points, method=method
            ).reshape(self.grid_lons.shape)
            
            # Fill NaN with nearest
            if np.isnan(self.u_field).any():
                u_nearest = griddata(points, u_values, grid_points, method='nearest')
                self.u_field = np.where(np.isnan(self.u_field), u_nearest.reshape(self.grid_lons.shape), self.u_field)
                
            if np.isnan(self.v_field).any():
                v_nearest = griddata(points, v_values, grid_points, method='nearest')
                self.v_field = np.where(np.isnan(self.v_field), v_nearest.reshape(self.grid_lons.shape), self.v_field)
                
        except Exception as e:
            logger.error(f"Interpolation failed: {e}")
    
    def get_flow_at_point(self, lat: float, lon: float) -> Optional[FlowVector]:
        """Get interpolated flow at a specific point."""
        if self.u_field is None or self.v_field is None:
            return None
        
        # Find nearest grid point
        lat_idx = int((lat - self.lat_bounds[0]) / self.resolution)
        lon_idx = int((lon - self.lon_bounds[0]) / self.resolution)
        
        if not (0 <= lat_idx < len(self.lats) and 0 <= lon_idx < len(self.lons)):
            return None
        
        u = self.u_field[lon_idx, lat_idx]
        v = self.v_field[lon_idx, lat_idx]
        
        magnitude = np.sqrt(u ** 2 + v ** 2)
        direction = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
        
        return FlowVector(
            latitude=lat,
            longitude=lon,
            u_component=u,
            v_component=v,
            magnitude=magnitude,
            direction=direction,
            timestamp=datetime.utcnow()
        )
    
    def compute_divergence(self) -> Optional[np.ndarray]:
        """Compute divergence of the flow field."""
        if self.u_field is None or self.v_field is None:
            return None
        
        # Approximate derivatives
        du_dx = np.gradient(self.u_field, self.resolution, axis=1)
        dv_dy = np.gradient(self.v_field, self.resolution, axis=0)
        
        return du_dx + dv_dy
    
    def compute_curl(self) -> Optional[np.ndarray]:
        """Compute curl (vorticity) of the flow field."""
        if self.u_field is None or self.v_field is None:
            return None
        
        # Approximate derivatives
        dv_dx = np.gradient(self.v_field, self.resolution, axis=1)
        du_dy = np.gradient(self.u_field, self.resolution, axis=0)
        
        return dv_dx - du_dy
    
    def to_dict(self) -> Dict[str, Any]:
        """Export flow field to dictionary for JSON serialization."""
        if self.u_field is None or self.v_field is None:
            return {}
        
        # Sample at lower resolution for visualization
        sample_rate = 4
        
        vectors = []
        for i in range(0, len(self.lons), sample_rate):
            for j in range(0, len(self.lats), sample_rate):
                u = self.u_field[i, j]
                v = self.v_field[i, j]
                
                if np.isnan(u) or np.isnan(v):
                    continue
                
                magnitude = np.sqrt(u ** 2 + v ** 2)
                direction = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
                
                vectors.append({
                    "lat": float(self.lats[j]),
                    "lon": float(self.lons[i]),
                    "u": round(float(u), 2),
                    "v": round(float(v), 2),
                    "magnitude": round(float(magnitude), 2),
                    "direction": round(float(direction), 1)
                })
        
        return {
            "bounds": {
                "lat": self.lat_bounds,
                "lon": self.lon_bounds
            },
            "resolution": self.resolution,
            "vectors": vectors
        }


class TemporalFlowAnalyzer:
    """
    Analyzes temporal patterns in weather data.
    """
    
    def __init__(self):
        pass
    
    def detect_periodicity(
        self,
        series: pd.Series,
        max_period: int = 168  # 1 week in hours
    ) -> Dict[str, Any]:
        """
        Detect periodic patterns in a time series.
        
        Args:
            series: Time series to analyze
            max_period: Maximum period to search for
        
        Returns:
            Dictionary with detected periods and their strengths
        """
        series = series.dropna()
        
        if len(series) < max_period * 2:
            return {"periods": [], "message": "Insufficient data"}
        
        # Autocorrelation
        values = series.values
        values = (values - values.mean()) / values.std()
        
        autocorr = correlate(values, values, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find peaks
        peaks, properties = find_peaks(
            autocorr[:max_period],
            height=0.1,
            distance=6
        )
        
        periods = []
        for peak, height in zip(peaks, properties.get('peak_heights', [])):
            periods.append({
                "period_hours": int(peak),
                "strength": round(float(height), 3),
                "interpretation": self._interpret_period(peak)
            })
        
        # Sort by strength
        periods.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            "periods": periods[:5],  # Top 5
            "dominant_period": periods[0] if periods else None
        }
    
    def _interpret_period(self, hours: int) -> str:
        """Interpret what a period might represent."""
        if 22 <= hours <= 26:
            return "Diurnal cycle (day/night)"
        elif 11 <= hours <= 13:
            return "Semi-diurnal (e.g., tidal influence)"
        elif 166 <= hours <= 170:
            return "Weekly pattern"
        elif hours == 12:
            return "Tidal cycle"
        else:
            return f"~{hours}h cycle"
    
    def detect_trend(
        self,
        series: pd.Series,
        window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Detect trends in the time series.
        
        Args:
            series: Time series
            window_hours: Window for trend calculation
        
        Returns:
            Trend information
        """
        series = series.dropna()
        
        if len(series) < window_hours:
            return {"trend": None}
        
        # Recent trend
        recent = series.tail(window_hours)
        
        if len(recent) < 2:
            return {"trend": None}
        
        x = np.arange(len(recent))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent.values)
        
        # Classify trend
        if p_value > 0.05:
            trend_type = "stable"
        elif slope > 0:
            trend_type = "increasing"
        else:
            trend_type = "decreasing"
        
        return {
            "trend": trend_type,
            "slope_per_hour": round(slope, 4),
            "r_squared": round(r_value ** 2, 3),
            "p_value": round(p_value, 4),
            "significant": p_value < 0.05,
            "change_24h": round(slope * 24, 2)
        }
    
    def detect_regime_changes(
        self,
        series: pd.Series,
        min_regime_hours: int = 12
    ) -> List[Dict[str, Any]]:
        """
        Detect regime changes (sudden shifts) in time series.
        
        Args:
            series: Time series
            min_regime_hours: Minimum regime duration
        
        Returns:
            List of detected regime changes
        """
        series = series.dropna()
        
        if len(series) < min_regime_hours * 3:
            return []
        
        # Compute rolling statistics
        roll_mean = series.rolling(min_regime_hours).mean()
        roll_std = series.rolling(min_regime_hours).std()
        
        # Detect changes using difference of means
        mean_diff = roll_mean.diff(min_regime_hours).abs()
        
        # Threshold: significant change is > 2 std
        threshold = 2 * roll_std.mean()
        
        changes = []
        in_change = False
        
        for idx, (timestamp, diff) in enumerate(mean_diff.items()):
            if pd.isna(diff):
                continue
            
            if diff > threshold and not in_change:
                changes.append({
                    "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                    "magnitude": round(float(diff), 2),
                    "before_mean": round(float(roll_mean.iloc[idx - min_regime_hours]), 2) if idx >= min_regime_hours else None,
                    "after_mean": round(float(roll_mean.iloc[idx]), 2)
                })
                in_change = True
            elif diff < threshold / 2:
                in_change = False
        
        return changes


class SpatialPatternDetector:
    """
    Detects spatial patterns across multiple stations.
    """
    
    def __init__(self, stations: Dict[str, Dict[str, Any]]):
        self.stations = stations
    
    def detect_gradient(
        self,
        values: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Detect spatial gradient in a variable across stations.
        
        Args:
            values: Dictionary mapping station IDs to values
        
        Returns:
            Gradient information
        """
        points = []
        vals = []
        
        for station_id, value in values.items():
            if station_id in self.stations and value is not None:
                st = self.stations[station_id]
                points.append([st.get('lon', st.get('longitude')), 
                              st.get('lat', st.get('latitude'))])
                vals.append(value)
        
        if len(points) < 3:
            return {"gradient": None}
        
        points = np.array(points)
        vals = np.array(vals)
        
        # Fit plane: v = a*lon + b*lat + c
        A = np.column_stack([points, np.ones(len(points))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, vals, rcond=None)
        except Exception:
            return {"gradient": None}
        
        # Gradient vector (in value per degree)
        grad_lon = coeffs[0]
        grad_lat = coeffs[1]
        
        # Direction of increasing values
        direction = (np.degrees(np.arctan2(grad_lon, grad_lat)) + 360) % 360
        magnitude = np.sqrt(grad_lon ** 2 + grad_lat ** 2)
        
        return {
            "gradient_direction": round(direction, 1),
            "gradient_magnitude": round(magnitude, 3),
            "increases_towards": self._direction_to_name(direction),
            "gradient_lon": round(grad_lon, 4),
            "gradient_lat": round(grad_lat, 4)
        }
    
    def _direction_to_name(self, bearing: float) -> str:
        """Convert bearing to compass direction."""
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        idx = int((bearing + 22.5) / 45) % 8
        return directions[idx]
    
    def find_spatial_anomalies(
        self,
        values: Dict[str, float],
        threshold_std: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Find stations with anomalous values compared to neighbors.
        
        Args:
            values: Station values
            threshold_std: Number of std devs for anomaly
        
        Returns:
            List of anomalous stations
        """
        if len(values) < 3:
            return []
        
        station_ids = list(values.keys())
        station_values = [values[s] for s in station_ids]
        
        mean_val = np.mean(station_values)
        std_val = np.std(station_values)
        
        if std_val < 0.001:
            return []
        
        anomalies = []
        
        for station_id, value in values.items():
            z_score = (value - mean_val) / std_val
            
            if abs(z_score) > threshold_std:
                anomalies.append({
                    "station_id": station_id,
                    "value": round(value, 2),
                    "z_score": round(z_score, 2),
                    "anomaly_type": "high" if z_score > 0 else "low"
                })
        
        return anomalies
