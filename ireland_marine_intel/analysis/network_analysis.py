"""
Network analysis for weather station correlations and flow patterns.
"""
import logging
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering, KMeans

logger = logging.getLogger(__name__)


@dataclass
class StationNode:
    """Represents a weather station as a network node."""
    station_id: str
    name: str
    latitude: float
    longitude: float
    station_type: str = "buoy"
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WeatherEdge:
    """Represents a connection between stations."""
    source: str
    target: str
    weight: float
    correlation: float
    lag: int = 0
    distance_km: float = 0
    variables: List[str] = field(default_factory=list)


class StationNetworkBuilder:
    """
    Builds and analyzes network graphs from weather station data.
    """
    
    def __init__(self):
        self.graph: Optional[nx.DiGraph] = None
        self.stations: Dict[str, StationNode] = {}
        
    def add_station(self, node: StationNode):
        """Add a station node."""
        self.stations[node.station_id] = node
    
    def build_correlation_network(
        self,
        data: Dict[str, pd.DataFrame],
        variable: str = "wind_speed",
        threshold: float = 0.5,
        max_lag: int = 24
    ) -> nx.DiGraph:
        """
        Build a directed network based on lagged correlations.
        
        Args:
            data: Dictionary mapping station IDs to DataFrames
            variable: Variable to correlate
            threshold: Minimum correlation to create edge
            max_lag: Maximum lag (hours) to consider
        
        Returns:
            Directed network graph
        """
        self.graph = nx.DiGraph()
        
        # Add nodes
        for station_id, node in self.stations.items():
            self.graph.add_node(
                station_id,
                name=node.name,
                lat=node.latitude,
                lon=node.longitude,
                type=node.station_type,
                **node.attributes
            )
        
        # Align all series
        aligned_data = {}
        for station_id, df in data.items():
            if variable in df.columns:
                series = df[variable].resample('1h').mean()
                aligned_data[station_id] = series
        
        station_ids = list(aligned_data.keys())
        
        # Compute lagged correlations
        for i, src in enumerate(station_ids):
            for j, tgt in enumerate(station_ids):
                if i == j:
                    continue
                
                src_series = aligned_data[src].dropna()
                tgt_series = aligned_data[tgt].dropna()
                
                # Align time indices
                common_idx = src_series.index.intersection(tgt_series.index)
                if len(common_idx) < max_lag * 4:
                    continue
                
                s1 = src_series.loc[common_idx]
                s2 = tgt_series.loc[common_idx]
                
                best_corr = 0
                best_lag = 0
                
                # Test positive lags (source leads target)
                for lag in range(0, max_lag + 1):
                    if lag == 0:
                        corr = s1.corr(s2)
                    else:
                        corr = s1.iloc[:-lag].reset_index(drop=True).corr(
                            s2.iloc[lag:].reset_index(drop=True)
                        )
                    
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
                
                # Only add edge if correlation exceeds threshold and lag > 0 (directional)
                if abs(best_corr) >= threshold and best_lag > 0:
                    distance = self._compute_distance(
                        self.stations[src],
                        self.stations[tgt]
                    )
                    
                    self.graph.add_edge(
                        src, tgt,
                        weight=abs(best_corr),
                        correlation=best_corr,
                        lag=best_lag,
                        distance_km=distance,
                        variable=variable
                    )
        
        logger.info(
            f"Built correlation network: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        
        return self.graph
    
    def build_proximity_network(
        self,
        max_distance_km: float = 200
    ) -> nx.Graph:
        """
        Build an undirected network based on geographic proximity.
        
        Args:
            max_distance_km: Maximum distance for connection
        
        Returns:
            Undirected network graph
        """
        graph = nx.Graph()
        
        # Add nodes
        for station_id, node in self.stations.items():
            graph.add_node(
                station_id,
                name=node.name,
                lat=node.latitude,
                lon=node.longitude,
                type=node.station_type
            )
        
        # Add edges based on distance
        station_ids = list(self.stations.keys())
        
        for i, src in enumerate(station_ids):
            for tgt in station_ids[i + 1:]:
                distance = self._compute_distance(
                    self.stations[src],
                    self.stations[tgt]
                )
                
                if distance <= max_distance_km:
                    # Weight inversely proportional to distance
                    weight = 1.0 / (1 + distance / 100)
                    
                    graph.add_edge(
                        src, tgt,
                        weight=weight,
                        distance_km=distance
                    )
        
        return graph
    
    def _compute_distance(self, node1: StationNode, node2: StationNode) -> float:
        """Compute haversine distance between two stations in km."""
        lat1 = np.radians(node1.latitude)
        lat2 = np.radians(node2.latitude)
        dlat = lat2 - lat1
        dlon = np.radians(node2.longitude - node1.longitude)
        
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 6371 * c  # Earth radius in km
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        if self.graph is None:
            return {}
        
        # Convert to undirected for some metrics
        undirected = self.graph.to_undirected()
        
        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
        }
        
        # Degree centrality
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())
        
        stats["in_degree_centrality"] = {k: v / max(1, self.graph.number_of_nodes() - 1) 
                                         for k, v in in_degree.items()}
        stats["out_degree_centrality"] = {k: v / max(1, self.graph.number_of_nodes() - 1) 
                                          for k, v in out_degree.items()}
        
        # Betweenness centrality
        if self.graph.number_of_edges() > 0:
            stats["betweenness_centrality"] = nx.betweenness_centrality(self.graph)
        
        # Strongly connected components
        sccs = list(nx.strongly_connected_components(self.graph))
        stats["num_strongly_connected_components"] = len(sccs)
        stats["largest_scc_size"] = max(len(c) for c in sccs) if sccs else 0
        
        return stats


class WeatherFlowAnalyzer:
    """
    Analyzes weather flow patterns across the station network.
    """
    
    def __init__(self, network: nx.DiGraph, stations: Dict[str, StationNode]):
        self.network = network
        self.stations = stations
    
    def identify_source_stations(self) -> List[str]:
        """
        Identify stations that act as weather "sources" (high out-degree, low in-degree).
        
        These are likely upwind stations that lead other stations.
        """
        scores = {}
        
        for node in self.network.nodes():
            in_deg = self.network.in_degree(node, weight='weight')
            out_deg = self.network.out_degree(node, weight='weight')
            
            # Source score: high out-degree relative to in-degree
            scores[node] = out_deg - in_deg
        
        # Return top sources
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [node for node, score in sorted_nodes if score > 0]
    
    def identify_sink_stations(self) -> List[str]:
        """
        Identify stations that act as weather "sinks" (high in-degree, low out-degree).
        
        These are likely downwind stations that follow other stations.
        """
        scores = {}
        
        for node in self.network.nodes():
            in_deg = self.network.in_degree(node, weight='weight')
            out_deg = self.network.out_degree(node, weight='weight')
            
            # Sink score: high in-degree relative to out-degree
            scores[node] = in_deg - out_deg
        
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [node for node, score in sorted_nodes if score > 0]
    
    def compute_influence_propagation(
        self,
        source_station: str,
        max_steps: int = 5
    ) -> Dict[str, List[Tuple[str, float, int]]]:
        """
        Compute how influence propagates from a source station.
        
        Returns paths showing how weather effects spread through the network.
        """
        if source_station not in self.network.nodes():
            return {}
        
        propagation = {source_station: [(source_station, 1.0, 0)]}
        visited = {source_station}
        current_level = {source_station}
        
        for step in range(1, max_steps + 1):
            next_level = set()
            
            for node in current_level:
                for neighbor in self.network.successors(node):
                    if neighbor in visited:
                        continue
                    
                    edge_data = self.network[node][neighbor]
                    influence = edge_data.get('weight', 0.5)
                    lag = edge_data.get('lag', 1)
                    
                    # Cumulative influence decays
                    parent_influence = propagation[node][-1][1]
                    cumulative_influence = parent_influence * influence * 0.9
                    
                    if neighbor not in propagation:
                        propagation[neighbor] = list(propagation[node])
                    
                    propagation[neighbor].append((neighbor, cumulative_influence, lag))
                    next_level.add(neighbor)
            
            visited.update(next_level)
            current_level = next_level
            
            if not current_level:
                break
        
        return propagation
    
    def infer_dominant_wind_direction(self) -> Dict[str, Any]:
        """
        Infer dominant wind direction from network structure.
        
        Assumes weather propagates with wind, so edges point downwind.
        """
        if self.network.number_of_edges() == 0:
            return {"direction": None, "confidence": 0}
        
        # Compute average vector of all edges
        vectors = []
        weights = []
        
        for src, tgt, data in self.network.edges(data=True):
            if src not in self.stations or tgt not in self.stations:
                continue
            
            src_node = self.stations[src]
            tgt_node = self.stations[tgt]
            
            # Direction vector (normalized)
            dlat = tgt_node.latitude - src_node.latitude
            dlon = tgt_node.longitude - src_node.longitude
            
            # Simple approximation (not accounting for spherical)
            length = np.sqrt(dlat ** 2 + dlon ** 2)
            if length > 0:
                vectors.append((dlon / length, dlat / length))
                weights.append(data.get('weight', 1))
        
        if not vectors:
            return {"direction": None, "confidence": 0}
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        avg_x = sum(v[0] * w for v, w in zip(vectors, weights))
        avg_y = sum(v[1] * w for v, w in zip(vectors, weights))
        
        # Convert to bearing
        bearing = np.degrees(np.arctan2(avg_x, avg_y)) % 360
        
        # Confidence based on consistency
        magnitude = np.sqrt(avg_x ** 2 + avg_y ** 2)
        
        return {
            "direction": round(bearing, 1),
            "direction_name": self._bearing_to_name(bearing),
            "confidence": round(magnitude, 2),
            "interpretation": f"Weather patterns generally propagate towards {self._bearing_to_name(bearing)}"
        }
    
    def _bearing_to_name(self, bearing: float) -> str:
        """Convert bearing to compass direction name."""
        directions = [
            "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
        ]
        idx = int((bearing + 11.25) / 22.5) % 16
        return directions[idx]


class ClusterAnalyzer:
    """
    Cluster stations into regions with similar weather patterns.
    """
    
    def __init__(self, stations: Dict[str, StationNode]):
        self.stations = stations
    
    def cluster_by_correlation(
        self,
        data: Dict[str, pd.DataFrame],
        variable: str = "wind_speed",
        n_clusters: int = 3
    ) -> Dict[str, int]:
        """
        Cluster stations by weather correlation patterns.
        
        Args:
            data: Station data
            variable: Variable to use for clustering
            n_clusters: Number of clusters
        
        Returns:
            Dictionary mapping station IDs to cluster labels
        """
        # Align data
        aligned = {}
        for station_id, df in data.items():
            if variable in df.columns:
                series = df[variable].resample('1h').mean()
                aligned[station_id] = series
        
        if len(aligned) < n_clusters:
            return {sid: 0 for sid in aligned.keys()}
        
        # Create correlation matrix
        station_ids = list(aligned.keys())
        n = len(station_ids)
        corr_matrix = np.zeros((n, n))
        
        for i, s1 in enumerate(station_ids):
            for j, s2 in enumerate(station_ids):
                common = aligned[s1].index.intersection(aligned[s2].index)
                if len(common) > 24:
                    corr = aligned[s1].loc[common].corr(aligned[s2].loc[common])
                    corr_matrix[i, j] = corr if not np.isnan(corr) else 0
                else:
                    corr_matrix[i, j] = 0 if i != j else 1
        
        # Convert to similarity/affinity matrix
        affinity = (corr_matrix + 1) / 2  # Scale to [0, 1]
        
        try:
            # Spectral clustering on affinity matrix
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42
            )
            labels = clustering.fit_predict(affinity)
        except Exception as e:
            logger.warning(f"Spectral clustering failed: {e}, using KMeans")
            # Fallback to geographic clustering
            return self.cluster_by_geography(n_clusters)
        
        return {station_ids[i]: int(labels[i]) for i in range(n)}
    
    def cluster_by_geography(self, n_clusters: int = 3) -> Dict[str, int]:
        """
        Cluster stations geographically using KMeans.
        
        Args:
            n_clusters: Number of clusters
        
        Returns:
            Dictionary mapping station IDs to cluster labels
        """
        if len(self.stations) < n_clusters:
            return {sid: 0 for sid in self.stations.keys()}
        
        station_ids = list(self.stations.keys())
        coords = np.array([
            [self.stations[sid].latitude, self.stations[sid].longitude]
            for sid in station_ids
        ])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(coords)
        
        return {station_ids[i]: int(labels[i]) for i in range(len(station_ids))}
    
    def get_cluster_summaries(
        self,
        cluster_assignments: Dict[str, int]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Summarize each cluster.
        
        Returns:
            Dictionary mapping cluster ID to summary
        """
        clusters: Dict[int, List[str]] = {}
        
        for station_id, cluster in cluster_assignments.items():
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(station_id)
        
        summaries = {}
        
        for cluster_id, station_ids in clusters.items():
            nodes = [self.stations[sid] for sid in station_ids if sid in self.stations]
            
            if not nodes:
                continue
            
            # Compute centroid
            avg_lat = np.mean([n.latitude for n in nodes])
            avg_lon = np.mean([n.longitude for n in nodes])
            
            # Determine region name based on position
            region = self._get_region_name(avg_lat, avg_lon)
            
            summaries[cluster_id] = {
                "stations": station_ids,
                "num_stations": len(station_ids),
                "centroid": {"lat": round(avg_lat, 4), "lon": round(avg_lon, 4)},
                "region": region,
                "station_types": list(set(n.station_type for n in nodes))
            }
        
        return summaries
    
    def _get_region_name(self, lat: float, lon: float) -> str:
        """Get approximate Irish region name from coordinates."""
        # Simplified Irish regions
        if lat > 54:
            if lon < -8:
                return "Northwest (Donegal/Sligo)"
            else:
                return "Northeast (Ulster)"
        elif lat > 53:
            if lon < -9:
                return "West (Connacht)"
            elif lon < -7:
                return "Midlands"
            else:
                return "East (Dublin/Leinster)"
        else:
            if lon < -9:
                return "Southwest (Kerry/Cork)"
            else:
                return "Southeast (Munster)"
