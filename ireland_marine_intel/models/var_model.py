"""
Vector Autoregressive (VAR) Model for weather forecasting.
"""
import logging
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class VARModelConfig:
    """Configuration for VAR model."""
    max_lags: int = 24
    ic: str = "aic"  # Information criterion: 'aic', 'bic', 'hqic'
    trend: str = "c"  # 'c' for constant, 'ct' for constant + trend, 'n' for none
    forecast_horizons: List[int] = None
    
    def __post_init__(self):
        if self.forecast_horizons is None:
            self.forecast_horizons = [6, 12, 24, 48, 72]


class VARForecaster:
    """
    Vector Autoregressive model for multivariate time series forecasting.
    
    Suitable for forecasting multiple weather variables simultaneously,
    capturing cross-correlations and lagged relationships.
    """
    
    def __init__(self, config: Optional[VARModelConfig] = None):
        self.config = config or VARModelConfig()
        self.model = None
        self.fitted_model = None
        self.scaler = StandardScaler()
        self.columns = None
        self.is_fitted = False
        self.optimal_lag = None
        
    def check_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """
        Check stationarity of a time series using ADF and KPSS tests.
        
        Args:
            series: Time series to test
        
        Returns:
            Dictionary with test results
        """
        series = series.dropna()
        
        if len(series) < 20:
            return {"stationary": None, "reason": "insufficient data"}
        
        # ADF test (null: unit root / non-stationary)
        try:
            adf_result = adfuller(series, autolag='AIC')
            adf_stationary = adf_result[1] < 0.05
        except Exception:
            adf_stationary = None
        
        # KPSS test (null: stationary)
        try:
            kpss_result = kpss(series, regression='c', nlags='auto')
            kpss_stationary = kpss_result[1] > 0.05
        except Exception:
            kpss_stationary = None
        
        return {
            "adf_stationary": adf_stationary,
            "kpss_stationary": kpss_stationary,
            "stationary": adf_stationary and kpss_stationary if (adf_stationary is not None and kpss_stationary is not None) else None
        }
    
    def difference_series(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Difference non-stationary series.
        
        Returns:
            Differenced DataFrame and dictionary of differencing orders
        """
        diff_orders = {}
        df_diff = df.copy()
        
        for col in df.columns:
            stationarity = self.check_stationarity(df[col])
            
            if stationarity["stationary"] == False:
                # Apply first-order differencing
                df_diff[col] = df[col].diff()
                diff_orders[col] = 1
                
                # Check if second differencing needed
                stationarity = self.check_stationarity(df_diff[col].dropna())
                if stationarity["stationary"] == False:
                    df_diff[col] = df_diff[col].diff()
                    diff_orders[col] = 2
            else:
                diff_orders[col] = 0
        
        return df_diff.dropna(), diff_orders
    
    def select_lag_order(self, df: pd.DataFrame) -> int:
        """
        Select optimal lag order using information criteria.
        
        Args:
            df: Preprocessed DataFrame
        
        Returns:
            Optimal lag order
        """
        try:
            model = VAR(df)
            lag_order_results = model.select_order(maxlags=min(self.config.max_lags, len(df) // 3))
            
            # Get lag based on selected criterion
            if self.config.ic == "aic":
                optimal_lag = lag_order_results.aic
            elif self.config.ic == "bic":
                optimal_lag = lag_order_results.bic
            else:
                optimal_lag = lag_order_results.hqic
            
            # Ensure at least 1 lag
            optimal_lag = max(1, optimal_lag)
            
            logger.info(f"Selected lag order: {optimal_lag} using {self.config.ic.upper()}")
            return optimal_lag
            
        except Exception as e:
            logger.warning(f"Lag selection failed: {e}. Using default lag=2")
            return 2
    
    def fit(
        self,
        df: pd.DataFrame,
        variables: Optional[List[str]] = None
    ) -> "VARForecaster":
        """
        Fit the VAR model to historical data.
        
        Args:
            df: DataFrame with datetime index and weather variables
            variables: Optional list of variables to include
        
        Returns:
            self
        """
        # Select variables
        if variables:
            df = df[variables].copy()
        else:
            df = df.select_dtypes(include=[np.number]).copy()
        
        self.columns = df.columns.tolist()
        
        # Handle missing values
        df = df.interpolate(method='time', limit=3)
        df = df.dropna()
        
        if len(df) < 48:
            raise ValueError("Insufficient data for VAR model (need at least 48 observations)")
        
        # Scale data
        scaled_data = self.scaler.fit_transform(df)
        df_scaled = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
        
        # Check and handle non-stationarity
        # For simplicity, we'll use the data as-is with trend component
        # In production, implement proper differencing
        
        # Select lag order
        self.optimal_lag = self.select_lag_order(df_scaled)
        
        # Fit VAR model
        self.model = VAR(df_scaled)
        self.fitted_model = self.model.fit(
            maxlags=self.optimal_lag,
            trend=self.config.trend
        )
        
        self.is_fitted = True
        logger.info(f"VAR model fitted with {len(self.columns)} variables, lag={self.optimal_lag}")
        
        return self
    
    def predict(
        self,
        steps: int = 24,
        return_conf_int: bool = True,
        alpha: float = 0.05
    ) -> Tuple[pd.DataFrame, Optional[Dict[str, Dict[str, pd.Series]]]]:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps ahead to forecast
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
        
        Returns:
            Tuple of (forecast DataFrame, confidence intervals dict)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")
        
        # Get forecast
        forecast_result = self.fitted_model.forecast(
            self.fitted_model.endog[-self.optimal_lag:],
            steps=steps
        )
        
        # Inverse transform
        forecast_unscaled = self.scaler.inverse_transform(forecast_result)
        
        # Create forecast index
        last_date = self.fitted_model.endog[-1]
        # Assume hourly data
        forecast_index = pd.date_range(
            start=pd.Timestamp.now(tz='UTC'),
            periods=steps,
            freq='1h'
        )
        
        forecast_df = pd.DataFrame(
            forecast_unscaled,
            index=forecast_index,
            columns=self.columns
        )
        
        # Confidence intervals
        confidence_intervals = None
        if return_conf_int:
            confidence_intervals = self._compute_confidence_intervals(
                forecast_result, steps, alpha
            )
        
        return forecast_df, confidence_intervals
    
    def _compute_confidence_intervals(
        self,
        forecast: np.ndarray,
        steps: int,
        alpha: float
    ) -> Dict[str, Dict[str, pd.Series]]:
        """Compute confidence intervals for forecasts."""
        try:
            # Get forecast error variance (approximation)
            sigma = self.fitted_model.sigma_u
            
            # Critical value
            z = stats.norm.ppf(1 - alpha / 2)
            
            confidence_intervals = {}
            
            for i, col in enumerate(self.columns):
                # Standard error increases with forecast horizon
                std_errors = np.sqrt(np.diag(sigma)[i] * np.arange(1, steps + 1))
                
                # Unscale
                scale = self.scaler.scale_[i]
                mean = self.scaler.mean_[i]
                
                forecast_col = forecast[:, i] * scale + mean
                std_errors_unscaled = std_errors * scale
                
                confidence_intervals[col] = {
                    "lower": pd.Series(forecast_col - z * std_errors_unscaled),
                    "upper": pd.Series(forecast_col + z * std_errors_unscaled)
                }
            
            return confidence_intervals
            
        except Exception as e:
            logger.warning(f"Failed to compute confidence intervals: {e}")
            return None
    
    def get_impulse_response(self, periods: int = 24) -> Dict[str, pd.DataFrame]:
        """
        Compute impulse response functions.
        
        Shows how a shock to one variable propagates through the system.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        irf = self.fitted_model.irf(periods=periods)
        
        results = {}
        for i, impulse_var in enumerate(self.columns):
            response_data = {}
            for j, response_var in enumerate(self.columns):
                response_data[response_var] = irf.irfs[:, j, i]
            
            results[impulse_var] = pd.DataFrame(response_data)
        
        return results
    
    def get_forecast_error_variance_decomposition(
        self,
        periods: int = 24
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute forecast error variance decomposition.
        
        Shows what fraction of forecast variance is explained by each variable.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        fevd = self.fitted_model.fevd(periods=periods)
        
        results = {}
        for i, var in enumerate(self.columns):
            results[var] = pd.DataFrame(
                fevd.decomp[:, i, :],
                columns=self.columns
            )
        
        return results
    
    def summary(self) -> str:
        """Get model summary."""
        if not self.is_fitted:
            return "Model not fitted"
        
        return str(self.fitted_model.summary())


class MultiStationForecaster:
    """
    Forecaster that handles multiple stations with aligned data.
    """
    
    def __init__(self):
        self.models: Dict[str, VARForecaster] = {}
        self.regional_model: Optional[VARForecaster] = None
    
    def fit_station_model(
        self,
        station_id: str,
        df: pd.DataFrame,
        variables: Optional[List[str]] = None
    ):
        """Fit a VAR model for a single station."""
        forecaster = VARForecaster()
        forecaster.fit(df, variables)
        self.models[station_id] = forecaster
        logger.info(f"Fitted model for station {station_id}")
    
    def fit_regional_model(
        self,
        data: Dict[str, pd.DataFrame],
        variable: str = "wind_speed"
    ):
        """
        Fit a regional VAR model using the same variable across stations.
        
        This captures spatial dependencies between stations.
        """
        # Align and combine data
        aligned_data = {}
        
        for station_id, df in data.items():
            if variable in df.columns:
                series = df[variable].resample('1h').mean()
                aligned_data[station_id] = series
        
        if len(aligned_data) < 2:
            raise ValueError("Need at least 2 stations for regional model")
        
        # Combine into single DataFrame
        combined = pd.DataFrame(aligned_data)
        combined = combined.dropna()
        
        # Fit model
        self.regional_model = VARForecaster()
        self.regional_model.fit(combined)
        logger.info(f"Fitted regional model with {len(aligned_data)} stations")
    
    def forecast_station(
        self,
        station_id: str,
        steps: int = 24
    ) -> Tuple[pd.DataFrame, Optional[Dict]]:
        """Generate forecast for a specific station."""
        if station_id not in self.models:
            raise ValueError(f"No model fitted for station {station_id}")
        
        return self.models[station_id].predict(steps)
    
    def forecast_regional(
        self,
        steps: int = 24
    ) -> Tuple[pd.DataFrame, Optional[Dict]]:
        """Generate regional forecast."""
        if self.regional_model is None:
            raise ValueError("Regional model not fitted")
        
        return self.regional_model.predict(steps)


class SimplePersistenceForecaster:
    """
    Simple persistence forecast with trend as baseline/fallback.
    """
    
    def __init__(self, trend_window: int = 24):
        self.trend_window = trend_window
        self.last_values = None
        self.trends = None
        self.stds = None
        self.columns = None
    
    def fit(self, df: pd.DataFrame):
        """Fit the persistence model."""
        df = df.select_dtypes(include=[np.number])
        self.columns = df.columns.tolist()
        
        self.last_values = df.iloc[-1].to_dict()
        
        # Compute trends from recent data
        recent = df.tail(self.trend_window)
        self.trends = {}
        self.stds = {}
        
        for col in self.columns:
            series = recent[col].dropna()
            if len(series) > 1:
                # Simple linear trend
                x = np.arange(len(series))
                slope, _ = np.polyfit(x, series.values, 1)
                self.trends[col] = slope
                self.stds[col] = series.std()
            else:
                self.trends[col] = 0
                self.stds[col] = 1
        
        return self
    
    def predict(self, steps: int = 24) -> Tuple[pd.DataFrame, Dict]:
        """Generate persistence forecasts."""
        forecasts = {}
        confidence_intervals = {}
        
        for col in self.columns:
            last = self.last_values.get(col, 0)
            trend = self.trends.get(col, 0)
            std = self.stds.get(col, 1)
            
            # Damped trend persistence
            values = []
            lowers = []
            uppers = []
            
            for h in range(1, steps + 1):
                damping = 0.95 ** h
                forecast = last + trend * h * damping
                
                # Uncertainty grows with horizon
                uncertainty = std * np.sqrt(h / 24) * 1.96
                
                values.append(forecast)
                lowers.append(forecast - uncertainty)
                uppers.append(forecast + uncertainty)
            
            forecasts[col] = values
            confidence_intervals[col] = {
                "lower": pd.Series(lowers),
                "upper": pd.Series(uppers)
            }
        
        forecast_df = pd.DataFrame(forecasts)
        
        return forecast_df, confidence_intervals
