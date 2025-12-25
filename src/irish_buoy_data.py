import pandas as pd
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
        return df[self.met_features + self.wave_features]