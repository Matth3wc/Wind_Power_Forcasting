"""
Model utilities for data preprocessing and feature engineering.
"""
import logging
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def prepare_time_series(
    df: pd.DataFrame,
    freq: str = "1h",
    interpolate_limit: int = 3
) -> pd.DataFrame:
    """
    Prepare time series data for modeling.
    
    Args:
        df: Raw DataFrame with datetime index
        freq: Target frequency
        interpolate_limit: Maximum consecutive NaNs to interpolate
    
    Returns:
        Preprocessed DataFrame
    """
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            raise ValueError("DataFrame must have datetime index or timestamp column")
    
    # Select numeric columns
    df = df.select_dtypes(include=[np.number]).copy()
    
    # Resample to regular frequency
    df = df.resample(freq).mean()
    
    # Interpolate small gaps
    df = df.interpolate(method='time', limit=interpolate_limit)
    
    # Forward fill remaining (up to limit)
    df = df.fillna(method='ffill', limit=interpolate_limit)
    
    return df


def compute_lagged_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 6, 12, 24],
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create lagged features for time series.
    
    Args:
        df: Input DataFrame
        lags: List of lag periods
        columns: Columns to create lags for (default: all)
    
    Returns:
        DataFrame with lagged features
    """
    if columns is None:
        columns = df.columns.tolist()
    
    result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        for lag in lags:
            result[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    return result


def compute_rolling_features(
    df: pd.DataFrame,
    windows: List[int] = [6, 12, 24],
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create rolling statistics features.
    
    Args:
        df: Input DataFrame
        windows: List of window sizes
        columns: Columns to compute for
    
    Returns:
        DataFrame with rolling features
    """
    if columns is None:
        columns = df.columns.tolist()
    
    result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        for window in windows:
            result[f"{col}_roll{window}_mean"] = df[col].rolling(window).mean()
            result[f"{col}_roll{window}_std"] = df[col].rolling(window).std()
            result[f"{col}_roll{window}_min"] = df[col].rolling(window).min()
            result[f"{col}_roll{window}_max"] = df[col].rolling(window).max()
    
    return result


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from datetime index.
    
    Args:
        df: DataFrame with datetime index
    
    Returns:
        DataFrame with temporal features
    """
    result = df.copy()
    
    result['hour'] = df.index.hour
    result['day_of_week'] = df.index.dayofweek
    result['day_of_year'] = df.index.dayofyear
    result['month'] = df.index.month
    
    # Cyclical encoding
    result['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    result['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    result['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    result['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    
    return result


def compute_wind_components(
    df: pd.DataFrame,
    speed_col: str = "wind_speed",
    direction_col: str = "wind_direction"
) -> pd.DataFrame:
    """
    Compute U and V wind components from speed and direction.
    
    Args:
        df: DataFrame with wind speed and direction
        speed_col: Name of wind speed column
        direction_col: Name of wind direction column
    
    Returns:
        DataFrame with wind components added
    """
    result = df.copy()
    
    if speed_col in df.columns and direction_col in df.columns:
        direction_rad = np.deg2rad(df[direction_col])
        result['wind_u'] = -df[speed_col] * np.sin(direction_rad)
        result['wind_v'] = -df[speed_col] * np.cos(direction_rad)
    
    return result


def detect_outliers(
    series: pd.Series,
    method: str = "iqr",
    threshold: float = 1.5
) -> pd.Series:
    """
    Detect outliers in a time series.
    
    Args:
        series: Input time series
        method: Detection method ('iqr', 'zscore', 'mad')
        threshold: Threshold for outlier detection
    
    Returns:
        Boolean series indicating outliers
    """
    series = series.dropna()
    
    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (series < lower) | (series > upper)
    
    elif method == "zscore":
        z_scores = np.abs(stats.zscore(series))
        return z_scores > threshold
    
    elif method == "mad":
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z = 0.6745 * (series - median) / mad
        return np.abs(modified_z) > threshold
    
    else:
        raise ValueError(f"Unknown method: {method}")


def align_multiple_series(
    data: Dict[str, pd.DataFrame],
    variable: str,
    freq: str = "1h"
) -> pd.DataFrame:
    """
    Align multiple time series from different sources.
    
    Args:
        data: Dictionary mapping source names to DataFrames
        variable: Variable to extract and align
        freq: Target frequency
    
    Returns:
        DataFrame with aligned series as columns
    """
    aligned = {}
    
    for source_name, df in data.items():
        if variable not in df.columns:
            continue
        
        series = df[variable].copy()
        
        # Resample
        series = series.resample(freq).mean()
        
        aligned[source_name] = series
    
    if not aligned:
        return pd.DataFrame()
    
    # Combine
    result = pd.DataFrame(aligned)
    
    # Find common time range
    result = result.dropna(how='all')
    
    return result


def compute_cross_correlations(
    df: pd.DataFrame,
    max_lag: int = 24
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Compute cross-correlations between all pairs of columns.
    
    Args:
        df: DataFrame with multiple columns
        max_lag: Maximum lag to consider
    
    Returns:
        Dictionary mapping column pairs to correlation info
    """
    columns = df.columns.tolist()
    results = {}
    
    for i, col1 in enumerate(columns):
        for col2 in columns[i + 1:]:
            series1 = df[col1].dropna()
            series2 = df[col2].dropna()
            
            # Align
            common_idx = series1.index.intersection(series2.index)
            if len(common_idx) < max_lag * 2:
                continue
            
            s1 = series1.loc[common_idx].values
            s2 = series2.loc[common_idx].values
            
            # Find best lag
            best_lag = 0
            best_corr = np.corrcoef(s1, s2)[0, 1]
            
            for lag in range(1, max_lag + 1):
                # s1 leads s2
                corr_pos = np.corrcoef(s1[:-lag], s2[lag:])[0, 1]
                if abs(corr_pos) > abs(best_corr):
                    best_corr = corr_pos
                    best_lag = lag
                
                # s2 leads s1
                corr_neg = np.corrcoef(s1[lag:], s2[:-lag])[0, 1]
                if abs(corr_neg) > abs(best_corr):
                    best_corr = corr_neg
                    best_lag = -lag
            
            results[(col1, col2)] = {
                "correlation": best_corr,
                "lag": best_lag,
                "leader": col1 if best_lag > 0 else col2
            }
    
    return results
