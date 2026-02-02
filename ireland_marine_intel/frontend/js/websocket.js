/**
 * Ireland Marine Weather Intelligence - WebSocket Module
 */

let ws = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY = 3000;

// Initialize WebSocket connection
function initializeWebSocket() {
    connectWebSocket();
}

// Connect to WebSocket server
function connectWebSocket() {
    try {
        ws = new WebSocket(CONFIG.WS_URL);
        
        ws.onopen = handleOpen;
        ws.onmessage = handleMessage;
        ws.onerror = handleError;
        ws.onclose = handleClose;
        
    } catch (error) {
        console.error('WebSocket connection error:', error);
        updateConnectionStatus('offline');
    }
}

// Handle WebSocket open
function handleOpen(event) {
    console.log('WebSocket connected');
    reconnectAttempts = 0;
    AppState.isConnected = true;
    updateConnectionStatus('online');
}

// Handle incoming messages
function handleMessage(event) {
    try {
        const message = JSON.parse(event.data);
        
        switch (message.type) {
            case 'initial_data':
                handleInitialData(message);
                break;
            
            case 'data_update':
                handleDataUpdate(message);
                break;
            
            case 'heartbeat':
                // Connection is alive
                break;
            
            default:
                console.log('Unknown message type:', message.type);
        }
        
    } catch (error) {
        console.error('Error parsing WebSocket message:', error);
    }
}

// Handle initial data message
function handleInitialData(message) {
    console.log('Received initial data');
    
    if (message.data) {
        AppState.latestData = message.data;
        updateMapMarkers();
        
        if (AppState.selectedStation) {
            updateStationDetails(AppState.selectedStation);
        }
    }
}

// Handle data update message
function handleDataUpdate(message) {
    console.log('Received data update');
    
    if (message.data) {
        // Merge new data with existing
        Object.assign(AppState.latestData, message.data);
        
        // Update UI
        updateMapMarkers();
        updateLastUpdateTime();
        
        if (AppState.selectedStation) {
            updateStationDetails(AppState.selectedStation);
        }
        
        // Show notification
        showToast('Live data updated', 'success');
    }
}

// Handle WebSocket error
function handleError(event) {
    console.error('WebSocket error:', event);
    updateConnectionStatus('offline');
}

// Handle WebSocket close
function handleClose(event) {
    console.log('WebSocket closed:', event.code, event.reason);
    AppState.isConnected = false;
    updateConnectionStatus('offline');
    
    // Attempt reconnection
    if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempts++;
        console.log(`Attempting reconnection ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS}...`);
        
        updateConnectionStatus('connecting');
        
        setTimeout(() => {
            connectWebSocket();
        }, RECONNECT_DELAY * reconnectAttempts);
    } else {
        console.log('Max reconnection attempts reached');
        showToast('Connection lost. Please refresh the page.', 'error');
    }
}

// Update connection status indicator
function updateConnectionStatus(status) {
    const statusDot = document.getElementById('connection-status');
    const statusText = document.getElementById('status-text');
    
    if (!statusDot || !statusText) return;
    
    // Remove all status classes
    statusDot.classList.remove('online', 'offline', 'connecting');
    
    switch (status) {
        case 'online':
            statusDot.classList.add('online');
            statusText.textContent = 'Connected';
            break;
        
        case 'offline':
            statusDot.classList.add('offline');
            statusText.textContent = 'Disconnected';
            break;
        
        case 'connecting':
            statusDot.classList.add('connecting');
            statusText.textContent = 'Reconnecting...';
            break;
    }
}

// Send message to server
function sendMessage(message) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
    } else {
        console.warn('WebSocket not connected, cannot send message');
    }
}

// Subscribe to specific station updates
function subscribeToStation(stationId) {
    sendMessage({
        type: 'subscribe',
        station_id: stationId
    });
}

// Unsubscribe from station updates
function unsubscribeFromStation(stationId) {
    sendMessage({
        type: 'unsubscribe',
        station_id: stationId
    });
}

// Request immediate data refresh
function requestRefresh() {
    sendMessage({
        type: 'refresh_request'
    });
}

// Close WebSocket connection
function closeWebSocket() {
    if (ws) {
        ws.close();
        ws = null;
    }
}

// Export functions
window.initializeWebSocket = initializeWebSocket;
window.sendMessage = sendMessage;
window.subscribeToStation = subscribeToStation;
window.unsubscribeFromStation = unsubscribeFromStation;
window.requestRefresh = requestRefresh;
window.closeWebSocket = closeWebSocket;

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    closeWebSocket();
});
