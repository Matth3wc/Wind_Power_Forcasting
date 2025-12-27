import streamlit as st
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to Python path
sys.path.insert(0, '/Users/mattthew/Documents/GitHub/Wind_Power_Forcasting/src')

from irish_buoy_data import IrishBuoyData

# Configure page
st.set_page_config(
    page_title="Irish Buoy Data Visualization",
    page_icon="🌊",
    layout="wide"
)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("muted")

# Title
st.title("🌊 Irish Wave Buoy Data Visualization")

# Sidebar controls
st.sidebar.title("Controls")
station = st.sidebar.selectbox(
    "Select Buoy Station", 
    ["M6", "M5", "M4", "M3", "M2", "M1"]
)
days = st.sidebar.slider("Days of Data", 7, 365, 30)

# Fetch data
with st.spinner(f'Fetching data from {station}...'):
    try:
        buoy = IrishBuoyData(station_id=station)
        data = buoy.fetch_data(days_back=days)
        st.success(f"✅ Loaded {len(data)} records from {station}")
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

# Display metrics
st.subheader("📊 Data Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(data))
col2.metric("Date Range", f"{(data.index.max() - data.index.min()).days} days")
col3.metric("Missing Values", data.isnull().sum().sum())
col4.metric("Sampling Rate", "Hourly")

# Show data sample
st.subheader("📋 Data Sample")
st.dataframe(data.head(10), use_container_width=True)

# Summary statistics
st.subheader("📈 Summary Statistics")
st.dataframe(data.describe(), use_container_width=True)

# Time Series Plots
st.subheader("📉 Time Series Visualizations")

# Wind and Temperature
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Wind Speed
axes[0, 0].plot(data.index, data['WindSpeed (knots)'], linewidth=1, color='steelblue')
axes[0, 0].set_title('Wind Speed Over Time', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Wind Speed (knots)')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# Air Temperature
axes[0, 1].plot(data.index, data['AirTemperature (degrees_C)'], linewidth=1, color='coral')
axes[0, 1].set_title('Air Temperature Over Time', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Temperature (°C)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# Atmospheric Pressure
axes[1, 0].plot(data.index, data['AtmosphericPressure (millibars)'], linewidth=1, color='green')
axes[1, 0].set_title('Atmospheric Pressure Over Time', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Pressure (mbar)')
axes[1, 0].set_xlabel('Date')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# Wave Heights
axes[1, 1].plot(data.index, data['WaveHeight (meters)'], linewidth=1, color='navy', label='Significant Wave Height')
axes[1, 1].plot(data.index, data['Hmax (meters)'], linewidth=1, color='red', alpha=0.6, label='Max Wave Height')
axes[1, 1].set_title('Wave Heights Over Time', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Height (m)')
axes[1, 1].set_xlabel('Date')
axes[1, 1].legend()
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# Wave Period
st.subheader("🌊 Wave Period Analysis")
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(data.index, data['Tp (seconds)'], linewidth=1, color='purple')
ax.set_title('Peak Wave Period Over Time', fontsize=12, fontweight='bold')
ax.set_ylabel('Period (seconds)')
ax.set_xlabel('Date')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# Distribution Plots
st.subheader("📊 Data Distributions")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Wind Speed Distribution
axes[0, 0].hist(data['WindSpeed (knots)'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Wind Speed Distribution')
axes[0, 0].set_xlabel('Wind Speed (knots)')
axes[0, 0].set_ylabel('Frequency')

# Air Temperature Distribution
axes[0, 1].hist(data['AirTemperature (degrees_C)'], bins=30, color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].set_title('Air Temperature Distribution')
axes[0, 1].set_xlabel('Temperature (°C)')
axes[0, 1].set_ylabel('Frequency')

# Pressure Distribution
axes[0, 2].hist(data['AtmosphericPressure (millibars)'], bins=30, color='green', alpha=0.7, edgecolor='black')
axes[0, 2].set_title('Atmospheric Pressure Distribution')
axes[0, 2].set_xlabel('Pressure (mbar)')
axes[0, 2].set_ylabel('Frequency')

# Wave Height Distribution
axes[1, 0].hist(data['WaveHeight (meters)'], bins=30, color='navy', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Wave Height Distribution')
axes[1, 0].set_xlabel('Height (m)')
axes[1, 0].set_ylabel('Frequency')

# Max Wave Height Distribution
axes[1, 1].hist(data['Hmax (meters)'], bins=30, color='red', alpha=0.7, edgecolor='black')
axes[1, 1].set_title('Max Wave Height Distribution')
axes[1, 1].set_xlabel('Height (m)')
axes[1, 1].set_ylabel('Frequency')

# Wave Period Distribution
axes[1, 2].hist(data['Tp (seconds)'], bins=30, color='purple', alpha=0.7, edgecolor='black')
axes[1, 2].set_title('Wave Period Distribution')
axes[1, 2].set_xlabel('Period (s)')
axes[1, 2].set_ylabel('Frequency')

plt.tight_layout()
st.pyplot(fig)

# Correlation Matrix
st.subheader("🔗 Correlation Analysis")

numeric_data = data.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Correlation Matrix - All Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
st.pyplot(fig)

# Key Statistics
st.subheader("📌 Key Statistics")

col1, col2 = st.columns(2)

with col1:
    st.write("**Wind & Weather**")
    st.write(f"- Avg Wind Speed: {data['WindSpeed (knots)'].mean():.2f} knots")
    st.write(f"- Max Wind Speed: {data['WindSpeed (knots)'].max():.2f} knots")
    st.write(f"- Min Wind Speed: {data['WindSpeed (knots)'].min():.2f} knots")
    st.write(f"- Avg Temperature: {data['AirTemperature (degrees_C)'].mean():.2f} °C")
    st.write(f"- Avg Pressure: {data['AtmosphericPressure (millibars)'].mean():.2f} mbar")

with col2:
    st.write("**Wave Conditions**")
    st.write(f"- Avg Wave Height: {data['WaveHeight (meters)'].mean():.2f} m")
    st.write(f"- Max Wave Height: {data['Hmax (meters)'].max():.2f} m")
    st.write(f"- Avg Wave Period: {data['Tp (seconds)'].mean():.2f} s")
    st.write(f"- Wave-Wind Correlation: {corr_matrix.loc['WaveHeight (meters)', 'WindSpeed (knots)']:.3f}")

# Footer
st.markdown("---")
st.markdown("**Data Source:** Irish Marine Institute ERDDAP Server | **Station:** " + station)