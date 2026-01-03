# check_exact_variables.py

from erddapy import ERDDAP
import pandas as pd
import requests
from io import StringIO

# Connect
e = ERDDAP(
    server="https://erddap.marine.ie/erddap",
    protocol="tabledap"
)
e.dataset_id = "IWBNetwork"

# Get info
info_url = e.get_info_url(response='csv')
response = requests.get(info_url)
df_info = pd.read_csv(StringIO(response.text))

# Get only variables (not attributes)
variables = df_info[df_info['Row Type'] == 'variable']['Variable Name'].tolist()

print("="*70)
print("ACTUAL AVAILABLE VARIABLES IN IRISH BUOY NETWORK")
print("="*70)
print(f"\nFound {len(variables)} variables:\n")

for var in variables:
    print(f"  • {var}")

# Save to file
with open('available_variables.txt', 'w') as f:
    for var in variables:
        f.write(f"{var}\n")

print("\n✓ Saved to 'available_variables.txt'")