/**
 * Ireland Marine Weather Intelligence - Charts Module
 */

// Initialize charts
function initializeCharts() {
    initializeHistoryChart();
    initializeForecastChart();
    console.log('Charts initialized');
}

// Initialize history chart
function initializeHistoryChart() {
    const ctx = document.getElementById('history-chart')?.getContext('2d');
    if (!ctx) return;
    
    AppState.historyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Wind Speed',
                data: [],
                borderColor: '#1a5f7a',
                backgroundColor: 'rgba(26, 95, 122, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleFont: { size: 12 },
                    bodyFont: { size: 11 },
                    padding: 10,
                    cornerRadius: 6
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'hour',
                        displayFormats: {
                            hour: 'HH:mm',
                            day: 'MMM d'
                        }
                    },
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 8
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            }
        }
    });
}

// Initialize forecast chart
function initializeForecastChart() {
    const ctx = document.getElementById('forecast-chart')?.getContext('2d');
    if (!ctx) return;
    
    AppState.forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Forecast',
                    data: [],
                    borderColor: '#57c5b6',
                    backgroundColor: 'rgba(87, 197, 182, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.4,
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: 'Upper Bound',
                    data: [],
                    borderColor: 'rgba(87, 197, 182, 0.3)',
                    backgroundColor: 'rgba(87, 197, 182, 0.1)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: '+1',
                    tension: 0.4,
                    pointRadius: 0
                },
                {
                    label: 'Lower Bound',
                    data: [],
                    borderColor: 'rgba(87, 197, 182, 0.3)',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleFont: { size: 12 },
                    bodyFont: { size: 11 },
                    padding: 10,
                    cornerRadius: 6,
                    filter: (item) => item.datasetIndex === 0
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'hour',
                        displayFormats: {
                            hour: 'HH:mm',
                            day: 'MMM d HH:mm'
                        }
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            }
        }
    });
}

// Update history chart with data
function updateHistoryChart(data, variable) {
    if (!AppState.historyChart || !data || data.length === 0) return;
    
    const chart = AppState.historyChart;
    
    // Filter valid data points
    const validData = data.filter(d => d[variable] !== null && d[variable] !== undefined);
    
    // Prepare data
    const labels = validData.map(d => new Date(d.timestamp));
    const values = validData.map(d => d[variable]);
    
    // Get variable info
    const varInfo = getVariableInfo(variable);
    
    // Update chart
    chart.data.labels = labels;
    chart.data.datasets[0].data = values;
    chart.data.datasets[0].label = varInfo.label;
    chart.data.datasets[0].borderColor = varInfo.color;
    chart.data.datasets[0].backgroundColor = varInfo.bgColor;
    
    // Update y-axis label
    chart.options.scales.y.title = {
        display: true,
        text: varInfo.unit
    };
    
    chart.update('none');
}

// Update forecast chart with data
function updateForecastChart(data) {
    if (!AppState.forecastChart || !data || !data.forecasts) return;
    
    const chart = AppState.forecastChart;
    const forecasts = data.forecasts;
    
    // Determine which variable to show (use first available)
    const variable = data.variables?.[0] || 'wind_speed';
    
    // Prepare data
    const labels = forecasts.map(f => new Date(f.timestamp));
    const values = forecasts.map(f => f[variable]);
    const upperBounds = forecasts.map(f => f[`${variable}_upper`] || f[variable] * 1.2);
    const lowerBounds = forecasts.map(f => f[`${variable}_lower`] || f[variable] * 0.8);
    
    // Get variable info
    const varInfo = getVariableInfo(variable);
    
    // Update chart data
    chart.data.labels = labels;
    chart.data.datasets[0].data = values;
    chart.data.datasets[0].label = `${varInfo.label} Forecast`;
    chart.data.datasets[0].borderColor = varInfo.color;
    
    chart.data.datasets[1].data = upperBounds;
    chart.data.datasets[2].data = lowerBounds;
    
    // Update y-axis
    chart.options.scales.y.title = {
        display: true,
        text: varInfo.unit
    };
    
    chart.update('none');
}

// Get variable display info
function getVariableInfo(variable) {
    const varMap = {
        'wind_speed': {
            label: 'Wind Speed',
            unit: 'knots',
            color: '#1a5f7a',
            bgColor: 'rgba(26, 95, 122, 0.1)'
        },
        'wave_height': {
            label: 'Wave Height',
            unit: 'meters',
            color: '#2980b9',
            bgColor: 'rgba(41, 128, 185, 0.1)'
        },
        'air_temperature': {
            label: 'Air Temperature',
            unit: '°C',
            color: '#e67e22',
            bgColor: 'rgba(230, 126, 34, 0.1)'
        },
        'sea_temperature': {
            label: 'Sea Temperature',
            unit: '°C',
            color: '#16a085',
            bgColor: 'rgba(22, 160, 133, 0.1)'
        },
        'air_pressure': {
            label: 'Atmospheric Pressure',
            unit: 'mb',
            color: '#8e44ad',
            bgColor: 'rgba(142, 68, 173, 0.1)'
        }
    };
    
    return varMap[variable] || {
        label: variable,
        unit: '',
        color: '#7f8c8d',
        bgColor: 'rgba(127, 140, 141, 0.1)'
    };
}

// Create mini sparkline chart
function createSparkline(containerId, data, color = '#1a5f7a') {
    const container = document.getElementById(containerId);
    if (!container || !data || data.length === 0) return;
    
    const width = container.offsetWidth;
    const height = 30;
    
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    container.innerHTML = '';
    container.appendChild(canvas);
    
    const ctx = canvas.getContext('2d');
    
    // Normalize data
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    
    // Draw line
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    
    data.forEach((value, i) => {
        const x = (i / (data.length - 1)) * width;
        const y = height - ((value - min) / range) * (height - 4) - 2;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    
    ctx.stroke();
}

// Export functions
window.initializeCharts = initializeCharts;
window.updateHistoryChart = updateHistoryChart;
window.updateForecastChart = updateForecastChart;
window.createSparkline = createSparkline;

// Add date-fns adapter for Chart.js time scale
// Include via CDN or bundler in production
if (typeof chartjsAdapterDateFns === 'undefined') {
    // Simple date formatting fallback
    Chart.defaults.scales.time = Chart.defaults.scales.time || {};
}
