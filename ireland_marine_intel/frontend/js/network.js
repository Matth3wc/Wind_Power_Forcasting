/**
 * Ireland Marine Weather Intelligence - Network Visualization Module
 */

let networkSvg = null;
let networkSimulation = null;

// Initialize network visualization
function initializeNetwork() {
    networkSvg = d3.select('#network-graph');
    
    // Initial fetch of network data
    fetchNetworkData();
    
    console.log('Network visualization initialized');
}

// Fetch network analysis data
async function fetchNetworkData() {
    const showCorrelations = document.getElementById('show-correlations')?.checked ?? true;
    const showDistances = document.getElementById('show-distances')?.checked ?? true;
    
    try {
        const response = await fetch(
            `${CONFIG.API_BASE_URL}/api/analysis/network?include_correlations=${showCorrelations}&include_distance=${showDistances}`
        );
        
        if (!response.ok) throw new Error('Failed to fetch network data');
        
        const data = await response.json();
        renderNetwork(data);
    } catch (error) {
        console.error('Error fetching network data:', error);
        // Render with local station data as fallback
        renderNetworkFallback();
    }
}

// Render the network graph
function renderNetwork(data) {
    if (!networkSvg || !data.nodes || !data.edges) return;
    
    // Clear existing
    networkSvg.selectAll('*').remove();
    
    const container = document.querySelector('.network-container');
    const width = container?.offsetWidth || 800;
    const height = 300;
    
    networkSvg
        .attr('width', width)
        .attr('height', height);
    
    // Create arrow marker for directed edges
    networkSvg.append('defs').append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '-0 -5 10 10')
        .attr('refX', 20)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('xoverflow', 'visible')
        .append('svg:path')
        .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
        .attr('fill', '#999')
        .style('stroke', 'none');
    
    // Prepare nodes and links for D3
    const nodes = data.nodes.map(n => ({
        id: n.station_id,
        name: n.name || n.station_id,
        type: n.station_type,
        lat: n.latitude,
        lon: n.longitude,
        cluster: n.cluster_id,
        centrality: n.centrality || 0
    }));
    
    const links = data.edges.map(e => ({
        source: e.source,
        target: e.target,
        weight: e.weight,
        type: e.edge_type,
        correlation: e.correlation,
        distance: e.distance_km
    }));
    
    // Create force simulation
    networkSimulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(d => {
            // Shorter distance for stronger connections
            return 100 - d.weight * 50;
        }))
        .force('charge', d3.forceManyBody().strength(-200))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(30));
    
    // Create links
    const link = networkSvg.append('g')
        .attr('class', 'links')
        .selectAll('line')
        .data(links)
        .enter()
        .append('line')
        .attr('class', d => `link ${d.type}`)
        .attr('stroke', d => d.type === 'correlation' ? '#57c5b6' : '#bdc3c7')
        .attr('stroke-width', d => Math.max(1, d.weight * 3))
        .attr('stroke-opacity', 0.6)
        .attr('stroke-dasharray', d => d.type === 'distance' ? '4,4' : 'none');
    
    // Create node groups
    const node = networkSvg.append('g')
        .attr('class', 'nodes')
        .selectAll('g')
        .data(nodes)
        .enter()
        .append('g')
        .attr('class', 'node')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
    
    // Add circles to nodes
    node.append('circle')
        .attr('r', d => 8 + d.centrality * 10)
        .attr('fill', d => getNodeColor(d.type))
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);
    
    // Add labels to nodes
    node.append('text')
        .attr('dx', 12)
        .attr('dy', 4)
        .text(d => d.id)
        .attr('font-size', '10px')
        .attr('fill', '#2c3e50');
    
    // Add tooltips
    node.append('title')
        .text(d => `${d.name}\nType: ${d.type}\nCentrality: ${d.centrality?.toFixed(3) || 'N/A'}`);
    
    // Click handler
    node.on('click', (event, d) => {
        selectStation(d.id);
    });
    
    // Update positions on tick
    networkSimulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        node
            .attr('transform', d => {
                // Keep nodes within bounds
                d.x = Math.max(20, Math.min(width - 20, d.x));
                d.y = Math.max(20, Math.min(height - 20, d.y));
                return `translate(${d.x},${d.y})`;
            });
    });
    
    // Add legend
    addNetworkLegend(networkSvg, width);
}

// Render network with fallback local data
function renderNetworkFallback() {
    if (!AppState.stations || Object.keys(AppState.stations).length === 0) return;
    
    const nodes = Object.entries(AppState.stations).map(([id, s]) => ({
        station_id: id,
        name: s.name,
        latitude: s.latitude,
        longitude: s.longitude,
        station_type: s.station_type
    }));
    
    // Create distance-based edges
    const edges = [];
    const stationIds = Object.keys(AppState.stations);
    
    for (let i = 0; i < stationIds.length; i++) {
        for (let j = i + 1; j < stationIds.length; j++) {
            const s1 = AppState.stations[stationIds[i]];
            const s2 = AppState.stations[stationIds[j]];
            
            // Calculate approximate distance
            const distance = haversineDistance(s1.latitude, s1.longitude, s2.latitude, s2.longitude);
            
            if (distance < 300) {
                edges.push({
                    source: stationIds[i],
                    target: stationIds[j],
                    weight: 1 / (1 + distance / 100),
                    distance_km: distance,
                    edge_type: 'distance'
                });
            }
        }
    }
    
    renderNetwork({ nodes, edges });
}

// Haversine distance calculation
function haversineDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // Earth's radius in km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
}

// Get node color based on type
function getNodeColor(type) {
    const colors = {
        'offshore_buoy': '#1a5f7a',
        'coastal_buoy': '#57c5b6',
        'lighthouse': '#f39c12',
        'synoptic': '#e74c3c',
        'observatory': '#9b59b6'
    };
    return colors[type] || '#7f8c8d';
}

// Add legend to network visualization
function addNetworkLegend(svg, width) {
    const legend = svg.append('g')
        .attr('class', 'legend')
        .attr('transform', `translate(${width - 150}, 20)`);
    
    const items = [
        { color: '#1a5f7a', label: 'Offshore Buoy' },
        { color: '#57c5b6', label: 'Coastal Buoy' },
        { color: '#f39c12', label: 'Lighthouse' }
    ];
    
    items.forEach((item, i) => {
        const g = legend.append('g')
            .attr('transform', `translate(0, ${i * 20})`);
        
        g.append('circle')
            .attr('r', 6)
            .attr('fill', item.color);
        
        g.append('text')
            .attr('x', 12)
            .attr('y', 4)
            .text(item.label)
            .attr('font-size', '10px')
            .attr('fill', '#2c3e50');
    });
}

// Drag handlers
function dragstarted(event, d) {
    if (!event.active) networkSimulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragended(event, d) {
    if (!event.active) networkSimulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

// Update network graph (called when checkboxes change)
function updateNetworkGraph() {
    fetchNetworkData();
}

// Export functions
window.initializeNetwork = initializeNetwork;
window.updateNetworkGraph = updateNetworkGraph;
window.fetchNetworkData = fetchNetworkData;
