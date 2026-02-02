/**
 * Ireland Marine Weather Intelligence - Map Module
 */

let map = null;
let markers = {};
let windArrows = [];

// Initialize the map
function initializeMap() {
    // Create map
    map = L.map('map', {
        center: CONFIG.MAP_CENTER,
        zoom: CONFIG.MAP_ZOOM,
        zoomControl: true,
        attributionControl: true
    });
    
    // Add tile layer (OpenStreetMap)
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 18
    }).addTo(map);
    
    // Add sea tiles for better marine visualization
    L.tileLayer('https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="http://www.openseamap.org">OpenSeaMap</a>',
        maxZoom: 18,
        opacity: 0.7
    }).addTo(map);
    
    // Add station markers
    addStationMarkers();
    
    // Update legend
    updateMapLegend('wind');
    
    console.log('Map initialized');
}

// Add markers for all stations
function addStationMarkers() {
    Object.entries(AppState.stations).forEach(([stationId, station]) => {
        const marker = createStationMarker(station);
        markers[stationId] = marker;
        marker.addTo(map);
    });
}

// Create a marker for a station
function createStationMarker(station) {
    const data = AppState.latestData[station.station_id];
    const color = data ? getWindSpeedColor(data.wind_speed) : '#7f8c8d';
    
    // Custom icon
    const icon = L.divIcon({
        className: 'custom-marker-container',
        html: `
            <div class="station-marker" style="background-color: ${color}">
                <span class="marker-label">${station.station_id}</span>
            </div>
        `,
        iconSize: [30, 30],
        iconAnchor: [15, 15],
        popupAnchor: [0, -20]
    });
    
    const marker = L.marker([station.latitude, station.longitude], { icon });
    
    // Create popup content
    const popupContent = createPopupContent(station, data);
    marker.bindPopup(popupContent);
    
    // Click handler
    marker.on('click', () => {
        selectStation(station.station_id);
    });
    
    return marker;
}

// Create popup content
function createPopupContent(station, data) {
    let content = `
        <div class="map-popup">
            <div class="popup-title">${station.name}</div>
            <div class="popup-type">${station.station_type.replace('_', ' ')}</div>
    `;
    
    if (data) {
        content += `
            <div class="popup-metrics">
                <div class="popup-metric">
                    <span>Wind:</span>
                    <span class="popup-metric-value">${data.wind_speed?.toFixed(1) || '--'} kts</span>
                </div>
                <div class="popup-metric">
                    <span>Wave:</span>
                    <span class="popup-metric-value">${data.wave_height?.toFixed(1) || '--'} m</span>
                </div>
                <div class="popup-metric">
                    <span>Temp:</span>
                    <span class="popup-metric-value">${data.air_temperature?.toFixed(1) || '--'} °C</span>
                </div>
                <div class="popup-metric">
                    <span>Pressure:</span>
                    <span class="popup-metric-value">${data.air_pressure?.toFixed(0) || '--'} mb</span>
                </div>
            </div>
        `;
        
        if (data.timestamp) {
            const updateTime = new Date(data.timestamp).toLocaleTimeString();
            content += `<div class="popup-time">Updated: ${updateTime}</div>`;
        }
    } else {
        content += '<div class="popup-offline">No recent data</div>';
    }
    
    content += '</div>';
    return content;
}

// Update map markers with latest data
function updateMapMarkers(displayVariable = null) {
    const variable = displayVariable || document.getElementById('map-layer-select')?.value || 'wind';
    
    // Clear existing wind arrows
    windArrows.forEach(arrow => map.removeLayer(arrow));
    windArrows = [];
    
    Object.entries(AppState.stations).forEach(([stationId, station]) => {
        const data = AppState.latestData[stationId];
        const marker = markers[stationId];
        
        if (!marker) return;
        
        // Determine color based on selected variable
        let color;
        switch (variable) {
            case 'wind':
                color = data ? getWindSpeedColor(data.wind_speed) : '#7f8c8d';
                break;
            case 'wave':
                color = data ? getWaveHeightColor(data.wave_height) : '#7f8c8d';
                break;
            case 'temperature':
                color = data ? getTemperatureColor(data.air_temperature) : '#7f8c8d';
                break;
            case 'pressure':
                color = data ? getPressureColor(data.air_pressure) : '#7f8c8d';
                break;
            default:
                color = '#7f8c8d';
        }
        
        // Update marker icon
        const icon = L.divIcon({
            className: 'custom-marker-container',
            html: `
                <div class="station-marker" style="background-color: ${color}">
                    <span class="marker-label">${stationId}</span>
                </div>
            `,
            iconSize: [30, 30],
            iconAnchor: [15, 15],
            popupAnchor: [0, -20]
        });
        
        marker.setIcon(icon);
        
        // Update popup
        const popupContent = createPopupContent(station, data);
        marker.setPopupContent(popupContent);
        
        // Add wind arrows for wind display
        if (variable === 'wind' && data && data.wind_direction !== null && data.wind_speed !== null) {
            addWindArrow(station, data);
        }
    });
    
    // Update legend
    updateMapLegend(variable);
}

// Add wind direction arrow
function addWindArrow(station, data) {
    const arrowLength = Math.min(0.3, data.wind_speed / 50);
    const direction = data.wind_direction;
    
    // Calculate arrow end point
    const endLat = station.latitude + arrowLength * Math.cos((direction - 180) * Math.PI / 180);
    const endLon = station.longitude + arrowLength * Math.sin((direction - 180) * Math.PI / 180) / Math.cos(station.latitude * Math.PI / 180);
    
    const arrow = L.polyline(
        [[station.latitude, station.longitude], [endLat, endLon]],
        {
            color: getWindSpeedColor(data.wind_speed),
            weight: 2,
            opacity: 0.8
        }
    );
    
    // Add arrowhead
    const arrowHead = L.polylineDecorator(arrow, {
        patterns: [{
            offset: '100%',
            repeat: 0,
            symbol: L.Symbol.arrowHead({
                pixelSize: 8,
                polygon: true,
                pathOptions: {
                    color: getWindSpeedColor(data.wind_speed),
                    fillOpacity: 1,
                    weight: 0
                }
            })
        }]
    });
    
    arrow.addTo(map);
    windArrows.push(arrow);
}

// Temperature color scale
function getTemperatureColor(temp) {
    if (temp === null || temp === undefined) return '#7f8c8d';
    if (temp < 5) return '#3498db';   // Cold - blue
    if (temp < 10) return '#2ecc71';  // Cool - green
    if (temp < 15) return '#f1c40f';  // Mild - yellow
    if (temp < 20) return '#e67e22';  // Warm - orange
    return '#e74c3c';  // Hot - red
}

// Pressure color scale
function getPressureColor(pressure) {
    if (pressure === null || pressure === undefined) return '#7f8c8d';
    if (pressure < 990) return '#e74c3c';   // Low - red (storm)
    if (pressure < 1000) return '#e67e22';  // Lowish - orange
    if (pressure < 1015) return '#f1c40f';  // Normal - yellow
    if (pressure < 1025) return '#2ecc71';  // High - green
    return '#3498db';  // Very high - blue
}

// Update map legend
function updateMapLegend(variable) {
    const legend = document.getElementById('map-legend');
    if (!legend) return;
    
    let legendItems = [];
    
    switch (variable) {
        case 'wind':
            legendItems = [
                { color: '#27ae60', label: '< 10 kts' },
                { color: '#2ecc71', label: '10-20 kts' },
                { color: '#f39c12', label: '20-30 kts' },
                { color: '#e67e22', label: '30-40 kts' },
                { color: '#e74c3c', label: '> 40 kts' }
            ];
            break;
        case 'wave':
            legendItems = [
                { color: '#3498db', label: '< 1m' },
                { color: '#2980b9', label: '1-2m' },
                { color: '#f39c12', label: '2-4m' },
                { color: '#e67e22', label: '4-6m' },
                { color: '#e74c3c', label: '> 6m' }
            ];
            break;
        case 'temperature':
            legendItems = [
                { color: '#3498db', label: '< 5°C' },
                { color: '#2ecc71', label: '5-10°C' },
                { color: '#f1c40f', label: '10-15°C' },
                { color: '#e67e22', label: '15-20°C' },
                { color: '#e74c3c', label: '> 20°C' }
            ];
            break;
        case 'pressure':
            legendItems = [
                { color: '#e74c3c', label: '< 990 mb' },
                { color: '#e67e22', label: '990-1000 mb' },
                { color: '#f1c40f', label: '1000-1015 mb' },
                { color: '#2ecc71', label: '1015-1025 mb' },
                { color: '#3498db', label: '> 1025 mb' }
            ];
            break;
    }
    
    legend.innerHTML = legendItems.map(item => `
        <div class="legend-item">
            <span class="legend-color" style="background-color: ${item.color}"></span>
            <span>${item.label}</span>
        </div>
    `).join('');
}

// Add custom CSS for markers
const markerStyles = document.createElement('style');
markerStyles.textContent = `
    .custom-marker-container {
        background: transparent !important;
        border: none !important;
    }
    
    .station-marker {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        border: 3px solid white;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        transition: transform 0.2s;
    }
    
    .station-marker:hover {
        transform: scale(1.2);
    }
    
    .marker-label {
        color: white;
        font-size: 8px;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .map-popup {
        min-width: 150px;
    }
    
    .popup-title {
        font-weight: 600;
        color: #1a5f7a;
        margin-bottom: 4px;
    }
    
    .popup-type {
        font-size: 0.8em;
        color: #7f8c8d;
        text-transform: capitalize;
        margin-bottom: 8px;
    }
    
    .popup-metrics {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 4px;
        font-size: 0.85em;
    }
    
    .popup-metric {
        display: flex;
        justify-content: space-between;
        padding: 2px 0;
    }
    
    .popup-metric-value {
        font-weight: 600;
    }
    
    .popup-time {
        margin-top: 8px;
        font-size: 0.75em;
        color: #95a5a6;
        text-align: right;
    }
    
    .popup-offline {
        color: #e74c3c;
        font-style: italic;
        padding: 8px 0;
    }
`;
document.head.appendChild(markerStyles);

// Export for global access
window.initializeMap = initializeMap;
window.updateMapMarkers = updateMapMarkers;
