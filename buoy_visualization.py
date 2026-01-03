import streamlit as st
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Add the src directory to Python path
sys.path.insert(0, '/Users/mattthew/Documents/GitHub/Wind_Power_Forcasting/src')

from irish_M_buoy_data import IrishBuoyData

# Configure page
st.set_page_config(
    page_title="Irish Wave Buoy Network",
    page_icon="🌊",
    layout="wide"
)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("muted")

BUOY_LOCATIONS = {
    # Offshore weather buoys (Irish Weather Buoy Network)
    "M2": {"lat": 53.4800, "lon": -5.4250, "name": "M2 Buoy"},
    "M3": {"lat": 51.2166, "lon": -10.5500, "name": "M3 Buoy"},
    "M4": {"lat": 55.0000, "lon": -10.0000, "name": "M4 Buoy"},
    "M5": {"lat": 51.6900, "lon": -6.7040, "name": "M5 Buoy"},
    "M6": {"lat": 53.0748, "lon": -15.8814, "name": "M6 Buoy"},

    # Coastal instrumented buoys (Irish Lights / coastal network)
    "IL1": {"lat": 52.5420, "lon": -9.7820, "name": "IL1 Buoy (Ballybunnion)"},
    "IL2": {"lat": 53.0470, "lon": -9.4850, "name": "IL2 Buoy (Finnis)"},
    "IL3": {"lat": 54.8780, "lon": -5.7550, "name": "IL3 Buoy (South Hunter)"},
    "IL4": {"lat": 52.2390, "lon": -6.2800, "name": "IL4 Buoy (Splaugh)"},
}


# Sidebar navigation
st.sidebar.title("🌊 Navigation")
main_page = st.sidebar.radio(
    "Select Page",
    ["Home", "History", "Wind Power Forecasting"]
)

# HOME PAGE
if main_page == "Home":
    # Minimal header
    st.markdown("<h1 style='text-align: center; color: #1f77b4;'>Irish Marine Buoy Network</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; color: #666;'>Real-time oceanographic and meteorological monitoring around Ireland</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Fetch current data for all buoys (last 24 hours)
    with st.spinner('Loading live buoy data...'):
        current_data = {}
        for buoy_id in BUOY_LOCATIONS.keys():
            try:
                buoy = IrishBuoyData(station_id=buoy_id)
                data = buoy.fetch_data(days_back=1)
                if len(data) > 0:
                    latest = data.iloc[-1]
                    current_data[buoy_id] = {
                        "wind_speed": latest['WindSpeed (knots)'],
                        "wave_height": latest['WaveHeight (meters)'],
                        "wave_period": latest['Tp (seconds)'],
                        "timestamp": data.index[-1]
                    }
            except:
                current_data[buoy_id] = None
    
    # Create interactive map with Plotly
    fig = go.Figure()
    
    # Add buoy markers
    for buoy_id, location in BUOY_LOCATIONS.items():
        if current_data.get(buoy_id):
            data_point = current_data[buoy_id]
            hover_text = f"""
            <b>{buoy_id}</b><br>
            Wind: {data_point['wind_speed']:.1f} kts<br>
            Wave: {data_point['wave_height']:.1f} m<br>
            Period: {data_point['wave_period']:.1f} s<br>
            <i>{data_point['timestamp'].strftime('%H:%M UTC')}</i>
            """
            marker_color = 'rgb(31, 119, 180)'  # Blue
        else:
            hover_text = f"<b>{buoy_id}</b><br><i>Offline</i>"
            marker_color = 'rgb(150, 150, 150)'  # Gray
        
        fig.add_trace(go.Scattergeo(
            lon=[location['lon']],
            lat=[location['lat']],
            text=buoy_id,
            mode='markers+text',
            marker=dict(
                size=18,
                color=marker_color,
                symbol='circle',
                line=dict(width=2, color='white')
            ),
            textposition="bottom center",
            textfont=dict(size=13, color='black', family='Arial Black'),
            hovertext=hover_text,
            hoverinfo='text',
            name=buoy_id,
            showlegend=False
        ))
    
    # Configure map with minimal style
    fig.update_geos(
        center=dict(lon=-8, lat=53),
        projection_scale=6.5,
        showcountries=True,
        countrycolor="rgba(200, 200, 200, 0.3)",
        showcoastlines=True,
        coastlinecolor="rgba(100, 100, 100, 0.5)",
        showland=True,
        landcolor="rgba(245, 245, 245, 1)",
        showocean=True,
        oceancolor="rgba(230, 245, 255, 1)",
        projection_type="mercator",
        bgcolor="rgba(0,0,0,0)"
    )
    
    fig.update_layout(
        height=550,
        margin={"r":0,"t":0,"l":0,"b":0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Minimal current conditions - compact cards
    st.markdown("<h3 style='text-align: center; margin-top: 30px;'>Current Conditions</h3>", unsafe_allow_html=True)
    
    cols = st.columns(5)
    for i, (buoy_id, data_point) in enumerate(current_data.items()):
        with cols[i]:
            if data_point:
                st.markdown(f"""
                <div style='text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 10px; border-left: 4px solid #1f77b4;'>
                    <h4 style='margin: 0; color: #1f77b4;'>{buoy_id}</h4>
                    <p style='margin: 5px 0; font-size: 24px; font-weight: bold; color: #333;'>{data_point['wind_speed']:.1f}</p>
                    <p style='margin: 0; font-size: 12px; color: #666;'>knots</p>
                    <hr style='margin: 10px 0; border: none; border-top: 1px solid #ddd;'>
                    <p style='margin: 5px 0; font-size: 16px; color: #555;'>{data_point['wave_height']:.1f}m</p>
                    <p style='margin: 0; font-size: 11px; color: #888;'>wave height</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 10px; border-left: 4px solid #999;'>
                    <h4 style='margin: 0; color: #999;'>{buoy_id}</h4>
                    <p style='margin: 10px 0; font-size: 14px; color: #999;'>Offline</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Minimal footer
    st.markdown("<p style='text-align: center; margin-top: 50px; font-size: 12px; color: #999;'>Data source: Marine Institute Ireland | Updated hourly</p>", unsafe_allow_html=True)

# HISTORY PAGE
elif main_page == "History":
    st.title("📊 Historical Buoy Data Analysis")
    
    # Sidebar controls for history
    st.sidebar.markdown("---")
    st.sidebar.subheader("Select Buoy Station")
    selected_buoy = st.sidebar.radio(
        "Station",
        ["M2", "M3", "M4", "M5", "M6"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Controls")
    days = st.sidebar.slider("Days of Historical Data", 7, 10000, 30)
    
    # Display buoy data (your existing function)
    def display_buoy_data(station, days):
        st.subheader(f"🌊 Buoy Station {station}")
        
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
        with st.expander("📋 View Data Sample"):
            st.dataframe(data.head(10), use_container_width=True)
        
        # Summary statistics
        with st.expander("📈 View Summary Statistics"):
            st.dataframe(data.describe(), use_container_width=True)
        
        # Time Series Plots
        st.subheader("📉 Time Series Visualizations")
        
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
        
        axes[0, 0].hist(data['WindSpeed (knots)'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Wind Speed Distribution')
        axes[0, 0].set_xlabel('Wind Speed (knots)')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].hist(data['AirTemperature (degrees_C)'], bins=50, color='coral', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Air Temperature Distribution')
        axes[0, 1].set_xlabel('Temperature (°C)')
        axes[0, 1].set_ylabel('Frequency')
        
        axes[0, 2].hist(data['AtmosphericPressure (millibars)'], bins=50, color='green', alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Atmospheric Pressure Distribution')
        axes[0, 2].set_xlabel('Pressure (mbar)')
        axes[0, 2].set_ylabel('Frequency')
        
        axes[1, 0].hist(data['WaveHeight (meters)'], bins=50, color='navy', alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Wave Height Distribution')
        axes[1, 0].set_xlabel('Height (m)')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].hist(data['Hmax (meters)'], bins=50, color='red', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Max Wave Height Distribution')
        axes[1, 1].set_xlabel('Height (m)')
        axes[1, 1].set_ylabel('Frequency')
        
        axes[1, 2].hist(data['Tp (seconds)'], bins=50, color='purple', alpha=0.7, edgecolor='black')
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
        
        st.markdown("---")
        st.markdown(f"**Data Source:** Irish Marine Institute ERDDAP Server | **Station:** {station}")
    
    display_buoy_data(selected_buoy, days)

# WIND POWER FORECASTING PAGE
elif main_page == "Wind Power Forecasting":
    st.title("⚡ Wind Power Forecasting")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Select Location")
    forecast_buoy = st.sidebar.radio(
        "Buoy Station",
        ["M2", "M3", "M4", "M5", "M6"]
    )
    
    st.subheader(f"Wind Power Forecast for {forecast_buoy}")
    
    st.info("🚧 **Coming Soon** - Wind power forecasting models are currently under development.")
    
    st.markdown("""
    ### Planned Features:
    
    - **Short-term Forecasting** (1-48 hours ahead)
        - LSTM neural network models
        - Baseline (meteorological-only) vs Wave-Enhanced models
        
    - **Model Comparison**
        - Performance metrics (RMSE, MAE, R²)
        - Forecast accuracy visualization
        
    - **Power Output Estimation**
        - Convert wind speed forecasts to power generation estimates
        - Account for turbine characteristics
        
    - **Real-time Updates**
        - Continuously updated forecasts as new data arrives
        - Confidence intervals and uncertainty quantification
    
    ### Current Status:
    - ✅ Data collection and visualization complete
    - ✅ Correlation analysis implemented
    - 🔄 Model training in progress
    - ⏳ Forecasting interface coming soon
    """)
    
    # Placeholder for future forecast visualization
    st.subheader("📈 Forecast Preview (Placeholder)")
    
    # Generate dummy forecast data for visualization
    import datetime
    future_dates = pd.date_range(start=datetime.datetime.now(), periods=48, freq='H')
    dummy_forecast = {
        'timestamp': future_dates,
        'wind_speed_forecast': np.random.uniform(10, 25, 48),
        'confidence_low': np.random.uniform(8, 15, 48),
        'confidence_high': np.random.uniform(20, 30, 48)
    }
    forecast_df = pd.DataFrame(dummy_forecast)
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(forecast_df['timestamp'], forecast_df['wind_speed_forecast'], 
            linewidth=2, color='blue', label='Forecast')
    ax.fill_between(forecast_df['timestamp'], 
                     forecast_df['confidence_low'], 
                     forecast_df['confidence_high'],
                     alpha=0.3, color='blue', label='Confidence Interval')
    ax.set_title('48-Hour Wind Speed Forecast (Demo)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Wind Speed (knots)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.caption("*This is demonstration data only. Actual forecasts will be available once models are trained.*")