# met_office_marine_data.py

import requests
import pandas as pd
import json
from typing import Dict, List, Optional
import os

class MetOfficeMarineData:
    """
    Class to fetch marine observation site data from Met Office DataPoint API.
    
    Requires an API key from: https://www.metoffice.gov.uk/services/data/datapoint
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize with API key.
        
        Args:
            api_key: Your Met Office DataPoint API key. 
                    If None, will try to read from environment variable MET_OFFICE_API_KEY
        """
        self.api_key = api_key or os.getenv('MET_OFFICE_API_KEY')
        
        if not self.api_key:
            raise ValueError("API key required. Either pass as argument or set MET_OFFICE_API_KEY environment variable")
        
        self.base_url = "http://datapoint.metoffice.gov.uk/public/data/val/wxmarineobs/all"
    
    def fetch_site_list_json(self) -> Dict:
        """
        Fetch marine observation sites as JSON.
        
        Returns:
            Dictionary containing site information
        """
        url = f"{self.base_url}/json/sitelist"
        params = {'key': self.api_key}
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def fetch_site_list_xml(self) -> str:
        """
        Fetch marine observation sites as XML.
        
        Returns:
            XML string containing site information
        """
        url = f"{self.base_url}/xml/sitelist"
        params = {'key': self.api_key}
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        return response.text
    
    def get_sites_dataframe(self) -> pd.DataFrame:
        """
        Get all marine observation sites as a pandas DataFrame.
        
        Returns:
            DataFrame with columns: id, name, latitude, longitude, obsLocationType, obsRegion, obsSource
        """
        data = self.fetch_site_list_json()
        
        locations = data['Locations']['Location']
        
        df = pd.DataFrame(locations)
        
        # Convert lat/lon to float
        df['latitude'] = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)
        
        return df
    
    def get_buoys_only(self) -> pd.DataFrame:
        """Get only buoy locations (excluding light vessels)."""
        df = self.get_sites_dataframe()
        return df[df['obsLocationType'] == 'Buoy']
    
    def get_irish_buoys(self) -> pd.DataFrame:
        """Get M-series buoys (M1-M6) used in Irish waters."""
        df = self.get_buoys_only()
        return df[df['name'].str.startswith('M') & df['name'].str[1].str.isdigit()]
    
    def get_site_by_name(self, name: str) -> Optional[Dict]:
        """
        Get specific site information by name.
        
        Args:
            name: Site name (e.g., 'M5')
        
        Returns:
            Dictionary with site information or None if not found
        """
        df = self.get_sites_dataframe()
        result = df[df['name'] == name]
        
        if len(result) == 0:
            return None
        
        return result.iloc[0].to_dict()
    
    def display_all_sites(self):
        """Display all sites in a formatted table."""
        df = self.get_sites_dataframe()
        
        print("="*80)
        print("MET OFFICE MARINE OBSERVATION SITES")
        print("="*80)
        print(f"\nTotal sites: {len(df)}")
        print(f"Buoys: {len(df[df['obsLocationType'] == 'Buoy'])}")
        print(f"Light Vessels: {len(df[df['obsLocationType'] == 'Light Vessel'])}")
        print("\n" + df.to_string(index=False))
    
    def display_by_region(self):
        """Display sites grouped by region."""
        df = self.get_sites_dataframe()
        
        print("="*80)
        print("SITES BY REGION")
        print("="*80)
        
        for region in df['obsRegion'].unique():
            print(f"\n{region}:")
            region_sites = df[df['obsRegion'] == region][['name', 'id', 'obsLocationType']]
            print(region_sites.to_string(index=False))
    
    def export_to_csv(self, filename: str = 'marine_sites.csv'):
        """Export site data to CSV file."""
        df = self.get_sites_dataframe()
        df.to_csv(filename, index=False)
        print(f"✓ Exported {len(df)} sites to {filename}")


# Example usage and demo
if __name__ == "__main__":
    # To use this, you need to:
    # 1. Get API key from: https://www.metoffice.gov.uk/services/data/datapoint
    # 2. Either:
    #    - Set environment variable: export MET_OFFICE_API_KEY="your_key_here"
    #    - Or pass directly: MetOfficeMarineData(api_key="your_key_here")
    
    try:
        # Initialize (will use environment variable MET_OFFICE_API_KEY)
        marine_data = MetOfficeMarineData()
        
        # Display all sites
        marine_data.display_all_sites()
        
        print("\n\n")
        
        # Display by region
        marine_data.display_by_region()
        
        print("\n\n")
        
        # Get Irish M-series buoys
        print("="*80)
        print("IRISH M-SERIES BUOYS")
        print("="*80)
        irish_buoys = marine_data.get_irish_buoys()
        print(irish_buoys.to_string(index=False))
        
        print("\n\n")
        
        # Get specific buoy
        print("="*80)
        print("SPECIFIC BUOY DETAILS - M5")
        print("="*80)
        m5_info = marine_data.get_site_by_name('M5')
        if m5_info:
            for key, value in m5_info.items():
                print(f"{key:20s}: {value}")
        
        # Export to CSV
        print("\n")
        marine_data.export_to_csv('marine_observation_sites.csv')
        
    except ValueError as e:
        print(f"ERROR: {e}")
        print("\nTo use this script:")
        print("1. Get API key from: https://www.metoffice.gov.uk/services/data/datapoint")
        print("2. Set environment variable:")
        print("   export MET_OFFICE_API_KEY='your_key_here'")
        print("\nOr pass API key directly:")
        print("   marine_data = MetOfficeMarineData(api_key='your_key_here')")
    
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print("Check that your API key is valid")