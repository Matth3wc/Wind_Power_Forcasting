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

# irish_M_buoy_data.py (FINAL CORRECTED VERSION)

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
        
        # ACTUAL AVAILABLE variables (verified from API)
        self.all_raw_variables = [
            "AtmosphericPressure",
            "WindDirection",
            "WindSpeed",
            "Gust",              # Note: "Gust" not "WindGust"
            "WaveHeight",
            "WavePeriod",
            "MeanWaveDirection",
            "Hmax",
            "AirTemperature",
            "DewPoint",
            "SeaTemperature",
            "salinity",          # Note: lowercase "salinity"
            "RelativeHumidity",
            "SprTp",
            "ThTp",
            "Tp"
        ]
        
        # Column names as they appear in DataFrame (with units)
        self.met_features = [
            "WindSpeed (knots)",
            "WindDirection (degrees)",
            "Gust (knots)",
            "AirTemperature (degrees_C)",
            "DewPoint (degrees_C)",
            "RelativeHumidity (%)",
            "AtmosphericPressure (millibars)"
        ]
        
        self.wave_features = [
            "WaveHeight (meters)",
            "Hmax (meters)",
            "WavePeriod (seconds)",
            "Tp (seconds)",
            "SprTp (seconds)",
            "ThTp (seconds)",
            "MeanWaveDirection (degrees)"
        ]
        
        self.water_features = [
            "SeaTemperature (degrees_C)",
            "salinity (PSU)"
        ]
        
        # Legacy basic features (for backward compatibility)
        self.basic_features = [
            "WindSpeed (knots)", 
            "AirTemperature (degrees_C)", 
            "AtmosphericPressure (millibars)",
            "WaveHeight (meters)", 
            "Hmax (meters)", 
            "Tp (seconds)"
        ]
        
    def fetch_data(self, days_back=30, variables: Optional[List[str]] = None, include_all=False):
        """
        Fetches data from the Marine Institute.
        
        Args:
            days_back: Number of days of historical data
            variables: List of raw variable names to fetch
            include_all: If True, fetch all available variables
        
        Returns:
            DataFrame with requested variables
        """
        if include_all:
            # Fetch all available variables
            request_vars = ["time", "station_id"] + self.all_raw_variables
        elif variables is not None:
            # Use specified variables
            request_vars = ["time", "station_id"] + variables
        else:
            # Default: use basic features for backward compatibility
            request_vars = ["time", "station_id", "WindSpeed", "AirTemperature", 
                           "AtmosphericPressure", "WaveHeight", "Hmax", "Tp"]
        
        self.e.variables = request_vars
        self.e.constraints = {
            "time>=": f"now-{days_back}days",
            "station_id=": self.station_id,
        }
        
        try:
            df = self.e.to_pandas(index_col="time (UTC)", parse_dates=True)
            return df.dropna()
        except Exception as e:
            print(f"Error: {e}")
            raise
    
    def fetch_all_data(self, days_back=30):
        """Fetch ALL available variables."""
        return self.fetch_data(days_back=days_back, include_all=True)
    
    def fetch_basic_data(self, days_back=30):
        """Fetch only basic 6 variables (backward compatible)."""
        return self.fetch_data(days_back=days_back, include_all=False)

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
        """Returns all features."""
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
        """Print all available columns in the DataFrame."""
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
        
        return df.columns.tolist()


# Test the corrected code
if __name__ == "__main__":
    print("="*70)
    print("IRISH BUOY DATA - COMPLETE ACCESS TEST")
    print("="*70)
    
    buoy = IrishBuoyData(station_id="M5")
    
    # Test 1: Basic data (backward compatible)
    print("\n1. Testing basic data (6 variables)...")
    df_basic = buoy.fetch_basic_data(days_back=7)
    print(f"   ✓ Fetched {len(df_basic)} records with {len(df_basic.columns)} columns")
    print(f"   Columns: {df_basic.columns.tolist()}")
    
    # Test 2: All available data
    print("\n2. Testing all available data (16 variables)...")
    df_all = buoy.fetch_all_data(days_back=7)
    print(f"   ✓ Fetched {len(df_all)} records with {len(df_all.columns)} columns")
    
    # List what's available
    buoy.list_available_columns(df_all)
    
    # Show sample
    print("\n" + "="*70)
    print("SAMPLE DATA - ALL VARIABLES (Latest 5 records)")
    print("="*70)
    print(df_all.tail())
    
    # Show comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Basic fetch:  {len(df_basic.columns)} variables")
    print(f"Complete fetch: {len(df_all.columns)} variables")
    print(f"\nYou were missing: {len(df_all.columns) - len(df_basic.columns)} variables!")
    
    missing_vars = [col for col in df_all.columns if col not in df_basic.columns]
    print(f"\nMissing variables were:")
    for var in missing_vars:
        print(f"  • {var}")
    
    # Test subsets
    print("\n" + "="*70)
    print("MET DATA")
    print("="*70)
    met_df = buoy.get_met_data(df_all)
    print(met_df.tail())
    
    print("\n" + "="*70)
    print("WAVE DATA")
    print("="*70)
    wave_df = buoy.get_wave_data(df_all)
    print(wave_df.tail())
    
    print("\n" + "="*70)
    print("WATER DATA")
    print("="*70)
    water_df = buoy.get_water_data(df_all)
    print(water_df.tail())
    
    # Save
    df_all.to_csv(f'buoy_{buoy.station_id}_complete.csv')
    print(f"\n✓ Saved complete data to 'buoy_{buoy.station_id}_complete.csv'")