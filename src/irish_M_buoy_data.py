'''import pandas as pd
from erddapy import ERDDAP

class IrishBuoyData:
    def __init__(self, station_id="M5"):
        self.station_id = station_id
        self.e = ERDDAP(
            server="https://erddap.marine.ie/erddap",
            protocol="tabledap",
        )
        self.e.dataset_id = "IWBNetwork"
        
        # Column names as they appear in the DataFrame (with units)
        self.met_features = [
            "WindSpeed (knots)", 
            "AirTemperature (degrees_C)", 
            "AtmosphericPressure (millibars)"
        ]
        self.wave_features = [
            "WaveHeight (meters)", 
            "Hmax (meters)", 
            "Tp (seconds)"
        ]
        
    def fetch_data(self, days_back=30):
        """Fetches the full dataset from the Marine Institute."""
        # Use raw variable names for the API request (no units)
        self.e.variables = ["time", "station_id", "WindSpeed", "AirTemperature", 
                           "AtmosphericPressure", "WaveHeight", "Hmax", "Tp"]
        self.e.constraints = {
            "time>=": f"now-{days_back}days",
            "station_id=": self.station_id,
        }
        
        df = self.e.to_pandas(index_col="time (UTC)", parse_dates=True)
        return df.dropna()

    def get_baseline_subset(self, df):
        """Returns only traditional meteorological factors."""
        return df[self.met_features]

    def get_wave_enhanced_subset(self, df):
        """Returns the wave-enhanced feature set."""
        return df[self.met_features + self.wave_features]'''


import pandas as pd
from erddapy import ERDDAP
from typing import List, Optional

class IrishBuoyData:
    def __init__(self, station_id="M5"):
        self.station_id = station_id
        self.e = ERDDAP(
            server="https://erddap.marine.ie/erddap",
            protocol="tabledap",
        )
        self.e.dataset_id = "IWBNetwork"
        
        # ALL available variables (raw names for API)
        self.all_raw_variables = [
            "WindSpeed",
            "WindDirection", 
            "WindGust",
            "AirTemperature",
            "DewPoint",
            "Humidity",
            "AtmosphericPressure",
            "WaveHeight",
            "Hmax",
            "MeanWavePeriod",
            "Tp",
            "PeakDirection",
            "MeanDirection",
            "SeaTemperature",
            "Salinity"
        ]
        
        # Column names as they appear in DataFrame (with units)
        self.met_features = [
            "WindSpeed (knots)", 
            "WindDirection (degrees)",
            "WindGust (knots)",
            "AirTemperature (degrees_C)",
            "DewPoint (degrees_C)",
            "Humidity (%)",
            "AtmosphericPressure (millibars)"
        ]
        
        self.wave_features = [
            "WaveHeight (meters)", 
            "Hmax (meters)",
            "MeanWavePeriod (seconds)",
            "Tp (seconds)",
            "PeakDirection (degrees)",
            "MeanDirection (degrees)"
        ]
        
        self.water_features = [
            "SeaTemperature (degrees_C)",
            "Salinity (PSU)"
        ]
        
        # Legacy features (for backward compatibility)
        self.basic_features = [
            "WindSpeed (knots)", 
            "AirTemperature (degrees_C)", 
            "AtmosphericPressure (millibars)",
            "WaveHeight (meters)", 
            "Hmax (meters)", 
            "Tp (seconds)"
        ]
        
    def fetch_data(self, days_back=30, variables: Optional[List[str]] = None):
        """
        Fetches data from the Marine Institute.
        
        Args:
            days_back: Number of days of historical data
            variables: List of raw variable names to fetch (None = all available)
        
        Returns:
            DataFrame with requested variables
        """
        # Use all variables if none specified
        if variables is None:
            request_vars = ["time", "station_id"] + self.all_raw_variables
        else:
            request_vars = ["time", "station_id"] + variables
        
        self.e.variables = request_vars
        self.e.constraints = {
            "time>=": f"now-{days_back}days",
            "station_id=": self.station_id,
        }
        
        try:
            df = self.e.to_pandas(index_col="time (UTC)", parse_dates=True)
            return df.dropna()
        except Exception as e:
            print(f"Warning: Some variables not available. Error: {e}")
            print("Falling back to basic variables...")
            
            # Fallback to basic variables
            basic_vars = ["time", "station_id", "WindSpeed", "AirTemperature", 
                         "AtmosphericPressure", "WaveHeight", "Hmax", "Tp"]
            self.e.variables = basic_vars
            df = self.e.to_pandas(index_col="time (UTC)", parse_dates=True)
            return df.dropna()
    
    def fetch_all_data(self, days_back=30):
        """Fetch all available variables."""
        return self.fetch_data(days_back=days_back, variables=None)
    
    def fetch_basic_data(self, days_back=30):
        """Fetch only the original 6 variables (for backward compatibility)."""
        basic_vars = ["WindSpeed", "AirTemperature", "AtmosphericPressure", 
                     "WaveHeight", "Hmax", "Tp"]
        return self.fetch_data(days_back=days_back, variables=basic_vars)

    def get_baseline_subset(self, df):
        """Returns only traditional meteorological factors."""
        available = [col for col in self.met_features if col in df.columns]
        return df[available]

    def get_wave_enhanced_subset(self, df):
        """Returns the wave-enhanced feature set."""
        all_features = self.met_features + self.wave_features
        available = [col for col in all_features if col in df.columns]
        return df[available]
    
    def get_all_features(self, df):
        """Returns all available features."""
        return df
    
    def get_met_data(self, df):
        """Returns only meteorological variables."""
        available = [col for col in self.met_features if col in df.columns]
        return df[available]
    
    def get_wave_data(self, df):
        """Returns only wave variables."""
        available = [col for col in self.wave_features if col in df.columns]
        return df[available]
    
    def get_water_data(self, df):
        """Returns water properties (temperature, salinity)."""
        available = [col for col in self.water_features if col in df.columns]
        return df[available]
    
    def list_available_columns(self, df):
        """Print all columns in the DataFrame."""
        print("="*70)
        print(f"AVAILABLE COLUMNS FOR {self.station_id}")
        print("="*70)
        
        print("\nMeteorological:")
        for feat in self.met_features:
            status = "✓" if feat in df.columns else "✗"
            print(f"  {status} {feat}")
        
        print("\nWave:")
        for feat in self.wave_features:
            status = "✓" if feat in df.columns else "✗"
            print(f"  {status} {feat}")
        
        print("\nWater:")
        for feat in self.water_features:
            status = "✓" if feat in df.columns else "✗"
            print(f"  {status} {feat}")
        
        print("\nOther:")
        other = [col for col in df.columns if col not in 
                (self.met_features + self.wave_features + self.water_features + ['station_id'])]
        for col in other:
            print(f"  ✓ {col}")
        
        print(f"\nTotal columns: {len(df.columns)}")

