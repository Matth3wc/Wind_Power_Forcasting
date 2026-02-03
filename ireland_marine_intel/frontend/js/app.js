/**
 * Ireland Marine Weather Intelligence - Main Application
 */

// Configuration
const CONFIG = {
    API_BASE_URL: window.location.protocol + '//' + window.location.hostname + ':8000',
    WS_URL: 'ws://' + window.location.hostname + ':8000/ws/live',
    REFRESH_INTERVAL: 60000, // 1 minute
    MAP_CENTER: [53.5, -8.0],
    MAP_ZOOM: 6
};

// Application State
const AppState = {
    stations: {},
    latestData: {},
    selectedStation: null,
    historyChart: null,
    forecastChart: null,
    wsConnection: null,
    isConnected: false
};

// Initialize application
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Initializing Ireland Marine Weather Intelligence...');
    
    try {
        // Initialize components
        await initializeStations();
        initializeMap();
        initializeCharts();
        initializeNetwork();
        initializeWebSocket();
        
        // Set up event listeners
        setupEventListeners();
        
        // Initial data fetch
        await fetchLatestData();
        
        // Start periodic refresh
        setInterval(fetchLatestData, CONFIG.REFRESH_INTERVAL);
        
        console.log('Application initialized successfully');
    } catch (error) {
        console.error('Initialization error:', error);
        showToast('Failed to initialize application', 'error');
    }
});

// Fetch station metadata
async function initializeStations() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/stations`);
        if (!response.ok) throw new Error('Failed to fetch stations');
        
        const stations = await response.json();
        
        stations.forEach(station => {
            AppState.stations[station.station_id] = station;
        });
        
        console.log(`Loaded ${stations.length} stations`);
    } catch (error) {
        console.error('Error fetching stations:', error);
        // Use fallback station data
        loadFallbackStations();
    }
}

// Fallback station data
function loadFallbackStations() {
    const fallbackStations = {
        // Offshore buoys (ERDDAP IWBNetwork)
        "M2": { station_id: "M2", name: "M2 Buoy", latitude: 53.48, longitude: -5.425, station_type: "offshore_buoy" },
        "M3": { station_id: "M3", name: "M3 Buoy", latitude: 51.2166, longitude: -10.55, station_type: "offshore_buoy" },
        "M4": { station_id: "M4", name: "M4 Buoy", latitude: 55.0, longitude: -10.0, station_type: "offshore_buoy" },
        "M5": { station_id: "M5", name: "M5 Buoy", latitude: 51.69, longitude: -6.704, station_type: "offshore_buoy" },
        "M6": { station_id: "M6", name: "M6 Buoy", latitude: 53.0748, longitude: -15.8814, station_type: "offshore_buoy" },
        // Met Éireann AWS stations
        "MalinHead": { station_id: "MalinHead", name: "Malin Head", latitude: 55.3719, longitude: -7.3389, station_type: "synoptic" },
        "Belmullet": { station_id: "Belmullet", name: "Belmullet", latitude: 54.2275, longitude: -10.0078, station_type: "synoptic" },
        "Valentia": { station_id: "Valentia", name: "Valentia Observatory", latitude: 51.9381, longitude: -10.2436, station_type: "observatory" },
        "RochesPoint": { station_id: "RochesPoint", name: "Roches Point", latitude: 51.7925, longitude: -8.2492, station_type: "lighthouse" },
        "MaceHead": { station_id: "MaceHead", name: "Mace Head", latitude: 53.3269, longitude: -9.8989, station_type: "observatory" },
        "SherkinIsland": { station_id: "SherkinIsland", name: "Sherkin Island", latitude: 51.4667, longitude: -9.4167, station_type: "coastal" }
    };
    
    AppState.stations = fallbackStations;
    console.log('Using fallback station data');
}

// Fetch latest weather data
async function fetchLatestData() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/weather/latest`);
        if (!response.ok) throw new Error('Failed to fetch weather data');
        
        const data = await response.json();
        AppState.latestData = data.stations || {};
        
        // Update UI
        updateMapMarkers();
        updateLastUpdateTime();
        
        if (AppState.selectedStation) {
            updateStationDetails(AppState.selectedStation);
        }
        
        console.log(`Updated data for ${Object.keys(AppState.latestData).length} stations`);
    } catch (error) {
        console.error('Error fetching latest data:', error);
    }
}

// Set up event listeners
function setupEventListeners() {
    // Refresh button
    document.getElementById('refresh-btn')?.addEventListener('click', async () => {
        await fetchLatestData();
        showToast('Data refreshed', 'success');
    });
    
    // Map layer select
    document.getElementById('map-layer-select')?.addEventListener('change', (e) => {
        updateMapMarkers(e.target.value);
    });
    
    // Chart variable select
    document.getElementById('chart-variable')?.addEventListener('change', () => {
        if (AppState.selectedStation) {
            fetchHistoricalData(AppState.selectedStation);
        }
    });
    
    // Chart period select
    document.getElementById('chart-period')?.addEventListener('change', () => {
        if (AppState.selectedStation) {
            fetchHistoricalData(AppState.selectedStation);
        }
    });
    
    // Network controls
    document.getElementById('show-correlations')?.addEventListener('change', updateNetworkGraph);
    document.getElementById('show-distances')?.addEventListener('change', updateNetworkGraph);
    
    // Analyze flows button
    document.getElementById('analyze-flows-btn')?.addEventListener('click', analyzeFlows);
}

// Update last update time
function updateLastUpdateTime() {
    const element = document.getElementById('last-update');
    if (element) {
        const now = new Date();
        element.textContent = `Last update: ${now.toLocaleTimeString()}`;
    }
}

// Station selection handler
function selectStation(stationId) {
    AppState.selectedStation = stationId;
    updateStationDetails(stationId);
    fetchHistoricalData(stationId);
    fetchForecast(stationId);
}

// Update station details panel
function updateStationDetails(stationId) {
    const station = AppState.stations[stationId];
    const data = AppState.latestData[stationId];
    
    if (!station) return;
    
    // Show details section
    document.querySelector('.placeholder-text')?.classList.add('hidden');
    document.getElementById('station-details')?.classList.remove('hidden');
    
    // Update station info
    document.getElementById('station-name').textContent = station.name;
    document.getElementById('station-type').textContent = station.station_type.replace('_', ' ');
    
    // Update metrics
    if (data) {
        document.getElementById('metric-wind').textContent = 
            data.wind_speed !== null ? data.wind_speed.toFixed(1) : '--';
        document.getElementById('metric-wave').textContent = 
            data.wave_height !== null ? data.wave_height.toFixed(1) : '--';
        document.getElementById('metric-temp').textContent = 
            data.air_temperature !== null ? data.air_temperature.toFixed(1) : '--';
        document.getElementById('metric-pressure').textContent = 
            data.air_pressure !== null ? data.air_pressure.toFixed(0) : '--';
        
        // Update wind compass
        if (data.wind_direction !== null) {
            drawWindCompass(data.wind_direction, data.wind_speed);
        }
    }
}

// Fetch historical data for a station
async function fetchHistoricalData(stationId) {
    const variable = document.getElementById('chart-variable')?.value || 'wind_speed';
    const hours = parseInt(document.getElementById('chart-period')?.value || '168');
    const days = Math.ceil(hours / 24);
    
    try {
        const response = await fetch(
            `${CONFIG.API_BASE_URL}/api/weather/${stationId}/history?days_back=${days}&resample_freq=1h`
        );
        
        if (!response.ok) {
            console.warn(`History not available for ${stationId}: ${response.status}`);
            return;
        }
        
        const data = await response.json();
        updateHistoryChart(data.data, variable);
    } catch (error) {
        console.error('Error fetching historical data:', error);
    }
}

// Fetch forecast for a station
async function fetchForecast(stationId) {
    try {
        const response = await fetch(
            `${CONFIG.API_BASE_URL}/api/forecasts/${stationId}?horizons=6&horizons=12&horizons=24&horizons=48&horizons=72`
        );
        
        if (!response.ok) {
            // Forecasts are only available for buoy stations
            console.warn(`Forecast not available for ${stationId}: ${response.status}`);
            return;
        }
        
        const data = await response.json();
        updateForecastChart(data);
    } catch (error) {
        console.error('Error fetching forecast:', error);
    }
}

// Analyze weather flows
async function analyzeFlows() {
    const variable = document.getElementById('chart-variable')?.value || 'wind_speed';
    
    try {
        const response = await fetch(
            `${CONFIG.API_BASE_URL}/api/analysis/flows?variable=${variable}`
        );
        
        if (!response.ok) throw new Error('Failed to fetch flow analysis');
        
        const data = await response.json();
        displayFlowPatterns(data);
    } catch (error) {
        console.error('Error analyzing flows:', error);
        showToast('Failed to analyze flows', 'error');
    }
}

// Display flow patterns
function displayFlowPatterns(data) {
    const container = document.getElementById('flow-patterns');
    const flowSection = document.getElementById('flow-analysis');
    
    if (!container || !flowSection) return;
    
    flowSection.classList.remove('hidden');
    container.innerHTML = '';
    
    if (data.patterns && data.patterns.length > 0) {
        data.patterns.slice(0, 5).forEach(pattern => {
            const div = document.createElement('div');
            div.className = 'flow-pattern';
            div.innerHTML = `
                <strong>${pattern.source_station}</strong> → <strong>${pattern.target_station}</strong>
                <br>Lag: ${pattern.lag_hours}h | Corr: ${pattern.correlation.toFixed(2)} | Dir: ${pattern.direction}
            `;
            container.appendChild(div);
        });
        
        if (data.dominant_flow_direction) {
            const summary = document.createElement('div');
            summary.className = 'flow-pattern';
            summary.innerHTML = `<strong>Dominant Flow:</strong> ${data.dominant_flow_direction}`;
            container.appendChild(summary);
        }
    } else {
        container.innerHTML = '<p>No significant flow patterns detected</p>';
    }
}

// Draw wind compass
function drawWindCompass(direction, speed) {
    const canvas = document.getElementById('wind-compass');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = 40;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw compass circle
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    ctx.strokeStyle = '#e0e6ed';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw cardinal directions
    ctx.fillStyle = '#7f8c8d';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('N', centerX, centerY - radius - 5);
    ctx.fillText('S', centerX, centerY + radius + 12);
    ctx.fillText('E', centerX + radius + 8, centerY + 3);
    ctx.fillText('W', centerX - radius - 8, centerY + 3);
    
    // Draw wind arrow
    const angleRad = (direction - 90) * Math.PI / 180;
    const arrowLength = radius * 0.8;
    
    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate(angleRad);
    
    // Arrow body
    ctx.beginPath();
    ctx.moveTo(-arrowLength * 0.3, 0);
    ctx.lineTo(arrowLength * 0.7, 0);
    ctx.strokeStyle = '#1a5f7a';
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Arrow head
    ctx.beginPath();
    ctx.moveTo(arrowLength * 0.7, 0);
    ctx.lineTo(arrowLength * 0.4, -8);
    ctx.lineTo(arrowLength * 0.4, 8);
    ctx.closePath();
    ctx.fillStyle = '#1a5f7a';
    ctx.fill();
    
    ctx.restore();
    
    // Draw speed text
    ctx.fillStyle = '#2c3e50';
    ctx.font = 'bold 12px Arial';
    ctx.fillText(`${speed?.toFixed(0) || '--'} kts`, centerX, centerY + 5);
}

// Show toast notification
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// Get wind speed color
function getWindSpeedColor(speed) {
    if (speed === null || speed === undefined) return '#7f8c8d';
    if (speed < 10) return '#27ae60';  // Calm - green
    if (speed < 20) return '#2ecc71';  // Light - light green
    if (speed < 30) return '#f39c12';  // Moderate - orange
    if (speed < 40) return '#e67e22';  // Fresh - dark orange
    return '#e74c3c';  // Strong - red
}

// Get wave height color
function getWaveHeightColor(height) {
    if (height === null || height === undefined) return '#7f8c8d';
    if (height < 1) return '#3498db';   // Small - light blue
    if (height < 2) return '#2980b9';   // Moderate - medium blue
    if (height < 4) return '#f39c12';   // Rough - orange
    if (height < 6) return '#e67e22';   // Very rough - dark orange
    return '#e74c3c';  // High - red
}

// Export for other modules
window.AppState = AppState;
window.CONFIG = CONFIG;
window.selectStation = selectStation;
window.showToast = showToast;
window.getWindSpeedColor = getWindSpeedColor;
window.getWaveHeightColor = getWaveHeightColor;
