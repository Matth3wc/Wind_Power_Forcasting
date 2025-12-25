# %% [markdown]
# # Import Libraries and Setup
# Import necessary libraries including pandas, numpy, matplotlib, seaborn, and the custom IrishBuoyData module. Configure plotting settings.

# %%
# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
import seaborn as sns  # For statistical data visualization
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.getcwd(), 'src')) # Custom module for fetching buoy data

# Configure plotting settings
plt.style.use('seaborn-v0_8-darkgrid')  # If you have matplotlib >= 3.6
sns.set_context("notebook", font_scale=1.2)  # Set context for seaborn plots with larger font scale
sns.set_palette("muted")  # Use a muted color palette for seaborn plots

# Set default figure size for matplotlib
plt.rcParams['figure.figsize'] = (10, 6)  # Default figure size is 10x6 inches

# Suppress scientific notation for pandas
pd.set_option('display.float_format', lambda x: f'{x:.3f}')  # Display floats with 3 decimal places

# %% [markdown]
# # Load and Explore Buoy Data
# Use IrishBuoyData class to fetch data from a selected buoy station (e.g., M6). Display data shape, date range, and sample records.

# %%
# Sidebar controls for selecting buoy station and days of data
station = st.sidebar.selectbox("Select Buoy Station", ["M6", "M5", "M4", "M3", "M2", "M1"])
days = st.sidebar.slider("Days of Data", 7, 365, 30)

# Fetch data using IrishBuoyData class
with st.spinner('Fetching data...'):
    buoy = IrishBuoyData(station_id=station)
    data = buoy.fetch_data(days_back=days)

# Display data shape and date range
st.success(f"✅ Loaded {len(data)} records from {station}")
col1, col2 = st.columns(2)
col1.metric("Total Records", len(data))
col2.metric("Date Range", f"{data.index.min().date()} to {data.index.max().date()}")

# Display sample records
st.subheader("Sample Records")
st.dataframe(data.head(10))

# %% [markdown]
# # Data Preprocessing and Cleaning
# Handle missing values, remove outliers, and ensure data quality. Check for gaps in time series data.

# %%
# Handle missing values by forward-filling and backward-filling
data = data.fillna(method='ffill').fillna(method='bfill')

# Remove outliers using the IQR method
Q1 = data.quantile(0.25)  # First quartile
Q3 = data.quantile(0.75)  # Third quartile
IQR = Q3 - Q1  # Interquartile range
outlier_mask = ~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)
data = data[outlier_mask]

# Check for gaps in time series data
time_diff = data.index.to_series().diff().dt.total_seconds()
gap_threshold = 3600  # Define a gap threshold of 1 hour
gaps = time_diff[time_diff > gap_threshold]

# Display gap information
if not gaps.empty:
    st.warning(f"⚠️ Detected {len(gaps)} gaps in the time series data exceeding {gap_threshold / 3600} hour(s).")
else:
    st.success("✅ No significant gaps detected in the time series data.")

# Display cleaned data summary
st.subheader("Cleaned Data Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Records (Cleaned)", len(data))
col2.metric("Missing Values (Cleaned)", data.isnull().sum().sum())
col3.metric("Outliers Removed", len(outlier_mask) - outlier_mask.sum())

# Display cleaned data sample
st.subheader("Cleaned Data Sample")
st.dataframe(data.head(10))

# %% [markdown]
# # Correlation Analysis
# Calculate and visualize correlation matrix between wind speed, wave height, Hmax, air temperature, and other parameters. Create heatmap and scatter plots.

# %%
# Calculate correlation matrix for numeric columns
numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
corr_matrix = numeric_data.corr()  # Compute correlation matrix

# Display correlation matrix as a heatmap
st.subheader("Correlation Matrix")
fig, ax = plt.subplots(figsize=(12, 8))  # Create a figure and axis
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)  # Plot heatmap
ax.set_title("Correlation Matrix Heatmap")  # Set title for the heatmap
st.pyplot(fig)  # Display the heatmap in Streamlit

# Display scatter plots for key relationships
st.subheader("Scatter Plots for Key Relationships")
col1, col2 = st.columns(2)  # Create two columns for scatter plots

# Scatter plot: Wind Speed vs Wave Height
with col1:
    fig, ax = plt.subplots()  # Create a figure and axis
    ax.scatter(data['WindSpeed (knots)'], data['WaveHeight (meters)'], alpha=0.6, s=20, color='blue')  # Scatter plot
    ax.set_xlabel("Wind Speed (knots)")  # Set x-axis label
    ax.set_ylabel("Wave Height (meters)")  # Set y-axis label
    ax.set_title("Wind Speed vs Wave Height")  # Set title
    st.pyplot(fig)  # Display the scatter plot in Streamlit

# Scatter plot: Wind Speed vs Hmax
with col2:
    fig, ax = plt.subplots()  # Create a figure and axis
    ax.scatter(data['WindSpeed (knots)'], data['Hmax (meters)'], alpha=0.6, s=20, color='red')  # Scatter plot
    ax.set_xlabel("Wind Speed (knots)")  # Set x-axis label
    ax.set_ylabel("Hmax (meters)")  # Set y-axis label
    ax.set_title("Wind Speed vs Hmax")  # Set title
    st.pyplot(fig)  # Display the scatter plot in Streamlit

# %% [markdown]
# # Time-Lagged Cross-Correlation
# Compute cross-correlation between wind speed and wave parameters at different time lags to identify optimal lag periods.

# %%
# Time-Lagged Cross-Correlation
st.subheader("⏳ Time-Lagged Cross-Correlation")

# Define a function to compute cross-correlation at different lags
def compute_cross_correlation(series1, series2, max_lag):
    lags = range(-max_lag, max_lag + 1)
    correlations = [series1.corr(series2.shift(lag)) for lag in lags]
    return lags, correlations

# Select parameters for cross-correlation analysis
parameter1 = st.sidebar.selectbox("Select Parameter 1", ["WindSpeed (knots)", "WaveHeight (meters)", "Hmax (meters)"])
parameter2 = st.sidebar.selectbox("Select Parameter 2", ["WaveHeight (meters)", "Hmax (meters)", "WindSpeed (knots)"])
max_lag = st.sidebar.slider("Max Lag (hours)", 1, 48, 12)

# Compute cross-correlation
with st.spinner("Computing cross-correlation..."):
    lags, correlations = compute_cross_correlation(data[parameter1], data[parameter2], max_lag)

# Plot cross-correlation
st.subheader(f"Cross-Correlation: {parameter1} vs {parameter2}")
fig, ax = plt.subplots()
ax.plot(lags, correlations, marker='o', linestyle='-', color='blue')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel("Lag (hours)")
ax.set_ylabel("Correlation Coefficient")
ax.set_title(f"Time-Lagged Cross-Correlation: {parameter1} vs {parameter2}")
ax.grid(True)
st.pyplot(fig)

# Display optimal lag
optimal_lag = lags[np.argmax(correlations)]
st.metric("Optimal Lag (hours)", optimal_lag)
st.metric("Max Correlation", f"{max(correlations):.3f}")

# %% [markdown]
# # Feature Engineering
# Create additional features such as lagged variables, rolling statistics, time-based features, and derived wave parameters.

# %%
# Feature Engineering

# Create lagged features for key parameters
lag_features = ['WindSpeed (knots)', 'WaveHeight (meters)', 'Hmax (meters)']
max_lag = 12  # Maximum lag in hours
for feature in lag_features:
    for lag in range(1, max_lag + 1):
        data[f'{feature}_lag{lag}'] = data[feature].shift(lag)

# Create rolling statistics (mean and standard deviation)
rolling_window = 6  # Rolling window size in hours
for feature in lag_features:
    data[f'{feature}_rolling_mean'] = data[feature].rolling(window=rolling_window).mean()
    data[f'{feature}_rolling_std'] = data[feature].rolling(window=rolling_window).std()

# Create time-based features
data['hour'] = data.index.hour  # Extract hour of the day
data['day_of_week'] = data.index.dayofweek  # Extract day of the week
data['month'] = data.index.month  # Extract month of the year

# Create derived wave parameters
data['WaveSteepness'] = data['WaveHeight (meters)'] / (data['WindSpeed (knots)'] + 1e-6)  # Avoid division by zero
data['WaveEnergy'] = 0.5 * 1025 * 9.81 * (data['WaveHeight (meters)'] ** 2)  # Energy per unit area (J/m²)

# Drop rows with NaN values introduced by lagging or rolling operations
data = data.dropna()

# Display engineered features
st.subheader("Engineered Features Sample")
st.dataframe(data.head(10))

# %% [markdown]
# # Train-Test Split
# Split the dataset into training and testing sets using temporal splitting to maintain time series integrity.

# %%
# Train-Test Split

# Define the split ratio for training and testing
train_ratio = 0.8  # 80% of the data will be used for training

# Calculate the split index based on the ratio
split_index = int(len(data) * train_ratio)

# Split the data into training and testing sets
train_data = data.iloc[:split_index]  # Training data
test_data = data.iloc[split_index:]  # Testing data

# Display the sizes of the training and testing sets
st.subheader("Train-Test Split Summary")
col1, col2 = st.columns(2)
col1.metric("Training Set Size", len(train_data))
col2.metric("Testing Set Size", len(test_data))

# Display the date ranges for training and testing sets
st.subheader("Train-Test Date Ranges")
col1, col2 = st.columns(2)
col1.metric("Training Set Date Range", f"{train_data.index.min().date()} to {train_data.index.max().date()}")
col2.metric("Testing Set Date Range", f"{test_data.index.min().date()} to {test_data.index.max().date()}")

# %% [markdown]
# # Baseline Model (Met-Only)
# Build and train a baseline forecasting model using only meteorological features (wind speed, air temperature, pressure). Implement LSTM or other time series model.

# %%
# Baseline Model (Met-Only)

# Import necessary libraries for model building
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Select meteorological features for the baseline model
met_features = ['WindSpeed (knots)', 'AirTemperature (degrees_C)', 'Pressure (hPa)']
target_feature = 'WindSpeed (knots)'

# Scale the features using MinMaxScaler
scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data[met_features])
scaled_test_data = scaler.transform(test_data[met_features])

# Prepare the data for LSTM
def create_sequences(data, target, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 24  # Use 24 hours of data for prediction
X_train, y_train = create_sequences(scaled_train_data, train_data[target_feature].values, sequence_length)
X_test, y_test = create_sequences(scaled_test_data, test_data[target_feature].values, sequence_length)

# Build the LSTM model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Plot training and validation loss
st.subheader("Training and Validation Loss")
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label='Training Loss')
ax.plot(history.history['val_loss'], label='Validation Loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss')
ax.legend()
st.pyplot(fig)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
st.metric("Test Loss (MSE)", f"{test_loss:.3f}")
st.metric("Test MAE", f"{test_mae:.3f}")

# Make predictions on the test set
predictions = model.predict(X_test)

# Plot actual vs predicted values
st.subheader("Actual vs Predicted Wind Speed")
fig, ax = plt.subplots()
ax.plot(test_data.index[sequence_length:], y_test, label='Actual', color='blue')
ax.plot(test_data.index[sequence_length:], predictions.flatten(), label='Predicted', color='orange')
ax.set_xlabel('Time')
ax.set_ylabel('Wind Speed (knots)')
ax.set_title('Actual vs Predicted Wind Speed')
ax.legend()
st.pyplot(fig)

# %% [markdown]
# # Wave-Enhanced Model
# Build and train an enhanced model that includes wave parameters (wave height, Hmax, wave period) in addition to meteorological features.

# %%
# Wave-Enhanced Model

# Select features for the wave-enhanced model
wave_features = ['WaveHeight (meters)', 'Hmax (meters)', 'WavePeriod (seconds)']
all_features = met_features + wave_features  # Combine meteorological and wave features

# Scale the features using MinMaxScaler
scaled_train_data = scaler.fit_transform(train_data[all_features])
scaled_test_data = scaler.transform(test_data[all_features])

# Prepare the data for LSTM
X_train, y_train = create_sequences(scaled_train_data, train_data[target_feature].values, sequence_length)
X_test, y_test = create_sequences(scaled_test_data, test_data[target_feature].values, sequence_length)

# Build the wave-enhanced LSTM model
wave_model = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dense(1)
])

# Compile the model
wave_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the wave-enhanced model with early stopping
wave_history = wave_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Plot training and validation loss for the wave-enhanced model
st.subheader("Wave-Enhanced Model: Training and Validation Loss")
fig, ax = plt.subplots()
ax.plot(wave_history.history['loss'], label='Training Loss')
ax.plot(wave_history.history['val_loss'], label='Validation Loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Wave-Enhanced Model: Training and Validation Loss')
ax.legend()
st.pyplot(fig)

# Evaluate the wave-enhanced model on the test set
wave_test_loss, wave_test_mae = wave_model.evaluate(X_test, y_test, verbose=0)
st.metric("Wave-Enhanced Model Test Loss (MSE)", f"{wave_test_loss:.3f}")
st.metric("Wave-Enhanced Model Test MAE", f"{wave_test_mae:.3f}")

# Make predictions on the test set using the wave-enhanced model
wave_predictions = wave_model.predict(X_test)

# Plot actual vs predicted values for the wave-enhanced model
st.subheader("Wave-Enhanced Model: Actual vs Predicted Wind Speed")
fig, ax = plt.subplots()
ax.plot(test_data.index[sequence_length:], y_test, label='Actual', color='blue')
ax.plot(test_data.index[sequence_length:], wave_predictions.flatten(), label='Predicted', color='green')
ax.set_xlabel('Time')
ax.set_ylabel('Wind Speed (knots)')
ax.set_title('Wave-Enhanced Model: Actual vs Predicted Wind Speed')
ax.legend()
st.pyplot(fig)

# %% [markdown]
# # Model Comparison and Evaluation
# Calculate and compare performance metrics (RMSE, MAE, R²) for both models. Perform statistical significance testing.

# %%
# Model Comparison and Evaluation

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ttest_rel

# Calculate performance metrics for the baseline model
baseline_rmse = np.sqrt(mean_squared_error(y_test, predictions))
baseline_mae = mean_absolute_error(y_test, predictions)
baseline_r2 = r2_score(y_test, predictions)

# Calculate performance metrics for the wave-enhanced model
wave_rmse = np.sqrt(mean_squared_error(y_test, wave_predictions))
wave_mae = mean_absolute_error(y_test, wave_predictions)
wave_r2 = r2_score(y_test, wave_predictions)

# Display performance metrics
st.subheader("Performance Metrics Comparison")
col1, col2 = st.columns(2)
with col1:
    st.metric("Baseline Model RMSE", f"{baseline_rmse:.3f}")
    st.metric("Baseline Model MAE", f"{baseline_mae:.3f}")
    st.metric("Baseline Model R²", f"{baseline_r2:.3f}")
with col2:
    st.metric("Wave-Enhanced Model RMSE", f"{wave_rmse:.3f}")
    st.metric("Wave-Enhanced Model MAE", f"{wave_mae:.3f}")
    st.metric("Wave-Enhanced Model R²", f"{wave_r2:.3f}")

# Perform statistical significance testing
st.subheader("Statistical Significance Testing")
t_stat, p_value = ttest_rel(predictions.flatten(), wave_predictions.flatten())

# Display t-test results
st.metric("t-statistic", f"{t_stat:.3f}")
st.metric("p-value", f"{p_value:.3e}")

# Interpretation of p-value
if p_value < 0.05:
    st.success("The difference in model performance is statistically significant (p < 0.05).")
else:
    st.warning("The difference in model performance is not statistically significant (p >= 0.05).")

# %% [markdown]
# # Visualization of Results
# Create plots showing actual vs predicted values, error distributions, and feature importance for both baseline and wave-enhanced models.

# %%
# Visualization of Results

# Plot actual vs predicted values for both models
st.subheader("Actual vs Predicted Wind Speed: Baseline vs Wave-Enhanced Models")
fig, ax = plt.subplots()
ax.plot(test_data.index[sequence_length:], y_test, label='Actual', color='blue', linewidth=1.5)
ax.plot(test_data.index[sequence_length:], predictions.flatten(), label='Baseline Predicted', color='orange', linestyle='--', linewidth=1.5)
ax.plot(test_data.index[sequence_length:], wave_predictions.flatten(), label='Wave-Enhanced Predicted', color='green', linestyle='--', linewidth=1.5)
ax.set_xlabel('Time')
ax.set_ylabel('Wind Speed (knots)')
ax.set_title('Actual vs Predicted Wind Speed')
ax.legend()
st.pyplot(fig)

# Plot error distributions for both models
st.subheader("Error Distributions: Baseline vs Wave-Enhanced Models")
baseline_errors = y_test - predictions.flatten()
wave_errors = y_test - wave_predictions.flatten()

fig, ax = plt.subplots()
sns.histplot(baseline_errors, kde=True, color='orange', label='Baseline Errors', ax=ax, bins=30)
sns.histplot(wave_errors, kde=True, color='green', label='Wave-Enhanced Errors', ax=ax, bins=30)
ax.set_xlabel('Error (knots)')
ax.set_ylabel('Frequency')
ax.set_title('Error Distributions')
ax.legend()
st.pyplot(fig)

# Plot feature importance for the wave-enhanced model
st.subheader("Feature Importance: Wave-Enhanced Model")
feature_importance = np.abs(wave_model.layers[0].get_weights()[0]).sum(axis=1)
feature_names = all_features

fig, ax = plt.subplots()
sns.barplot(x=feature_importance, y=feature_names, palette='viridis', ax=ax)
ax.set_xlabel('Importance')
ax.set_ylabel('Features')
ax.set_title('Feature Importance: Wave-Enhanced Model')
st.pyplot(fig)


