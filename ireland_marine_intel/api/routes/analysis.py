"""
Network and mesh analysis API routes.
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query
import numpy as np

from config.settings import ALL_STATIONS, BUOY_STATIONS, COASTAL_STATIONS
from api.models.schemas import (
    NetworkAnalysisResponse, NetworkNode, NetworkEdge,
    FlowAnalysisResponse, FlowPattern, BathymetryInfo
)
from ingestion.buoy_fetcher import BuoyFetcher
from ingestion.bathymetry_fetcher import BathymetryFetcher, BathymetryAnalyzer

router = APIRouter()


@router.get("/network")
async def get_network_analysis(
    include_correlations: bool = Query(True, description="Include correlation-based edges"),
    include_distance: bool = Query(True, description="Include distance-based edges"),
    correlation_threshold: float = Query(0.5, ge=0, le=1),
    max_distance_km: float = Query(300, ge=0)
) -> Dict[str, Any]:
    """
    Get network analysis of the station system.
    
    Returns nodes (stations) and edges (connections based on correlation
    and/or geographic distance).
    """
    from geopy.distance import geodesic
    
    # Build nodes
    nodes = []
    station_list = {**BUOY_STATIONS, **COASTAL_STATIONS}
    
    for station_id, info in station_list.items():
        nodes.append({
            "station_id": station_id,
            "name": info["name"],
            "latitude": info["lat"],
            "longitude": info["lon"],
            "station_type": info["type"],
        })
    
    # Build edges
    edges = []
    station_ids = list(station_list.keys())
    
    # Distance-based edges
    if include_distance:
        for i, sid1 in enumerate(station_ids):
            info1 = station_list[sid1]
            coord1 = (info1["lat"], info1["lon"])
            
            for sid2 in station_ids[i + 1:]:
                info2 = station_list[sid2]
                coord2 = (info2["lat"], info2["lon"])
                
                distance = geodesic(coord1, coord2).kilometers
                
                if distance <= max_distance_km:
                    # Weight inversely proportional to distance
                    weight = 1 / (1 + distance / 100)
                    
                    edges.append({
                        "source": sid1,
                        "target": sid2,
                        "weight": round(weight, 3),
                        "distance_km": round(distance, 1),
                        "edge_type": "distance"
                    })
    
    # Correlation-based edges (requires data fetch)
    correlation_matrix = None
    if include_correlations:
        try:
            fetcher = BuoyFetcher()
            data = await fetcher.fetch_all_buoys(days_back=7)
            
            # Compute correlations for wind_speed
            if len(data) >= 2:
                from analysis.correlation_network import CorrelationNetworkBuilder
                
                builder = CorrelationNetworkBuilder()
                correlation_matrix = builder.compute_correlation_matrix(
                    data, variable="wind_speed"
                )
                
                # Add correlation edges
                for i, sid1 in enumerate(correlation_matrix.index):
                    for sid2 in correlation_matrix.columns[i + 1:]:
                        corr = correlation_matrix.loc[sid1, sid2]
                        
                        if abs(corr) >= correlation_threshold:
                            edges.append({
                                "source": sid1,
                                "target": sid2,
                                "weight": round(abs(corr), 3),
                                "correlation": round(corr, 3),
                                "edge_type": "correlation"
                            })
        except Exception as e:
            # Continue without correlation edges
            pass
    
    # Compute simple network statistics
    from collections import defaultdict
    
    degree = defaultdict(int)
    for edge in edges:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1
    
    # Identify clusters (simplified)
    clusters = {}
    # For now, cluster by station type
    for i, node in enumerate(nodes):
        station_type = node["station_type"]
        cluster_id = hash(station_type) % 5
        node["cluster_id"] = cluster_id
        
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node["station_id"])
    
    # Compute centrality
    for node in nodes:
        node["degree"] = degree.get(node["station_id"], 0)
        node["centrality"] = round(degree.get(node["station_id"], 0) / max(len(nodes) - 1, 1), 3)
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "clusters": clusters,
        "statistics": {
            "avg_degree": round(sum(degree.values()) / len(degree) if degree else 0, 2),
            "max_degree": max(degree.values()) if degree else 0,
            "density": round(2 * len(edges) / (len(nodes) * (len(nodes) - 1)) if len(nodes) > 1 else 0, 3),
            "n_clusters": len(clusters),
        }
    }


@router.get("/flows")
async def get_flow_analysis(
    variable: str = Query("wind_speed", description="Variable to analyze"),
    max_lag_hours: int = Query(12, ge=1, le=48, description="Maximum lag to consider")
) -> Dict[str, Any]:
    """
    Analyze weather flow patterns between stations.
    
    Identifies time-lagged correlations that suggest weather system
    propagation between stations.
    """
    fetcher = BuoyFetcher()
    
    try:
        data = await fetcher.fetch_all_buoys(days_back=7)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch data: {str(e)}")
    
    if len(data) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 stations for flow analysis")
    
    # Check that variable exists
    valid_stations = {}
    for station_id, df in data.items():
        if variable in df.columns:
            valid_stations[station_id] = df[variable].resample("1h").mean().interpolate(limit=3)
    
    if len(valid_stations) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Variable '{variable}' not available in enough stations"
        )
    
    # Compute cross-correlations with lags
    from scipy import stats
    
    patterns = []
    station_ids = list(valid_stations.keys())
    
    for i, sid1 in enumerate(station_ids):
        series1 = valid_stations[sid1].dropna()
        
        for sid2 in station_ids[i + 1:]:
            series2 = valid_stations[sid2].dropna()
            
            # Align series
            common_idx = series1.index.intersection(series2.index)
            if len(common_idx) < max_lag_hours * 2:
                continue
            
            s1 = series1.loc[common_idx].values
            s2 = series2.loc[common_idx].values
            
            # Test different lags
            best_lag = 0
            best_corr = stats.pearsonr(s1, s2)[0]
            
            for lag in range(1, max_lag_hours + 1):
                # s1 leads s2 by lag hours
                corr_pos = stats.pearsonr(s1[:-lag], s2[lag:])[0] if lag < len(s1) else 0
                # s2 leads s1 by lag hours
                corr_neg = stats.pearsonr(s1[lag:], s2[:-lag])[0] if lag < len(s1) else 0
                
                if abs(corr_pos) > abs(best_corr):
                    best_corr = corr_pos
                    best_lag = lag  # Positive: s1 leads
                
                if abs(corr_neg) > abs(best_corr):
                    best_corr = corr_neg
                    best_lag = -lag  # Negative: s2 leads
            
            # Only report significant patterns
            if abs(best_corr) >= 0.4:
                # Determine flow direction
                info1 = ALL_STATIONS[sid1]
                info2 = ALL_STATIONS[sid2]
                
                if best_lag > 0:
                    source, target = sid1, sid2
                    direction = _compute_direction(
                        info1["lat"], info1["lon"],
                        info2["lat"], info2["lon"]
                    )
                else:
                    source, target = sid2, sid1
                    direction = _compute_direction(
                        info2["lat"], info2["lon"],
                        info1["lat"], info1["lon"]
                    )
                
                patterns.append({
                    "source_station": source,
                    "target_station": target,
                    "lag_hours": abs(best_lag),
                    "correlation": round(best_corr, 3),
                    "direction": direction,
                    "variable": variable
                })
    
    # Sort by correlation strength
    patterns.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    
    # Determine dominant flow direction
    if patterns:
        directions = [p["direction"] for p in patterns[:5]]
        direction_counts = {}
        for d in directions:
            # Simplify to cardinal direction
            cardinal = _to_cardinal(d)
            direction_counts[cardinal] = direction_counts.get(cardinal, 0) + 1
        
        dominant = max(direction_counts, key=direction_counts.get)
    else:
        dominant = "undetermined"
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "variable": variable,
        "max_lag_hours": max_lag_hours,
        "pattern_count": len(patterns),
        "patterns": patterns,
        "dominant_flow_direction": dominant,
        "flow_speed_estimate_kmh": _estimate_flow_speed(patterns) if patterns else None
    }


@router.get("/clusters")
async def get_cluster_analysis(
    n_clusters: int = Query(3, ge=2, le=10, description="Number of clusters"),
    method: str = Query("correlation", enum=["correlation", "geographic", "combined"])
) -> Dict[str, Any]:
    """
    Perform cluster analysis on stations.
    
    Groups stations based on weather pattern similarity, geographic
    proximity, or a combination.
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from geopy.distance import geodesic
    
    station_list = {**BUOY_STATIONS, **COASTAL_STATIONS}
    station_ids = list(station_list.keys())
    n_stations = len(station_ids)
    
    if n_stations < n_clusters:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot create {n_clusters} clusters from {n_stations} stations"
        )
    
    # Build feature matrix based on method
    if method == "geographic":
        # Use lat/lon as features
        features = np.array([
            [station_list[sid]["lat"], station_list[sid]["lon"]]
            for sid in station_ids
        ])
        
        # Normalize
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        
    elif method == "correlation":
        # Use correlation matrix as features
        fetcher = BuoyFetcher()
        
        try:
            data = asyncio.run(fetcher.fetch_all_buoys(days_back=7))
        except:
            # Fallback to geographic
            features = np.array([
                [station_list[sid]["lat"], station_list[sid]["lon"]]
                for sid in station_ids
            ])
            features = (features - features.mean(axis=0)) / features.std(axis=0)
        else:
            # Compute pairwise correlations
            features = np.eye(n_stations)  # Start with identity
            
            for i, sid1 in enumerate(station_ids):
                if sid1 not in data:
                    continue
                df1 = data[sid1]
                if "wind_speed" not in df1.columns:
                    continue
                s1 = df1["wind_speed"].resample("1h").mean()
                
                for j, sid2 in enumerate(station_ids[i + 1:], i + 1):
                    if sid2 not in data:
                        continue
                    df2 = data[sid2]
                    if "wind_speed" not in df2.columns:
                        continue
                    s2 = df2["wind_speed"].resample("1h").mean()
                    
                    # Compute correlation
                    common = s1.index.intersection(s2.index)
                    if len(common) > 10:
                        corr = s1.loc[common].corr(s2.loc[common])
                        features[i, j] = corr
                        features[j, i] = corr
    
    else:  # combined
        # Combine geographic and type information
        type_map = {"offshore_buoy": 0, "coastal_buoy": 1}
        features = np.array([
            [
                station_list[sid]["lat"],
                station_list[sid]["lon"],
                type_map.get(station_list[sid]["type"], 2)
            ]
            for sid in station_ids
        ])
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
    
    # Perform clustering
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(features)
    
    # Build result
    clusters = {}
    for i, sid in enumerate(station_ids):
        cluster_id = int(labels[i])
        if cluster_id not in clusters:
            clusters[cluster_id] = {
                "stations": [],
                "center_lat": 0,
                "center_lon": 0,
            }
        
        clusters[cluster_id]["stations"].append(sid)
        clusters[cluster_id]["center_lat"] += station_list[sid]["lat"]
        clusters[cluster_id]["center_lon"] += station_list[sid]["lon"]
    
    # Compute cluster centers
    for cluster_id, cluster_info in clusters.items():
        n = len(cluster_info["stations"])
        cluster_info["center_lat"] = round(cluster_info["center_lat"] / n, 4)
        cluster_info["center_lon"] = round(cluster_info["center_lon"] / n, 4)
        cluster_info["size"] = n
    
    # Station assignments
    assignments = {
        sid: int(labels[i])
        for i, sid in enumerate(station_ids)
    }
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "method": method,
        "n_clusters": n_clusters,
        "clusters": clusters,
        "assignments": assignments
    }


@router.get("/bathymetry")
async def get_bathymetry_analysis() -> Dict[str, Any]:
    """
    Get bathymetry analysis for all stations.
    
    Returns depth information and exposure analysis for each station.
    """
    analyzer = BathymetryAnalyzer()
    
    station_list = {**BUOY_STATIONS, **COASTAL_STATIONS}
    results = {}
    
    for station_id in station_list:
        try:
            analysis = await analyzer.analyze_station_bathymetry(station_id)
            results[station_id] = analysis
        except Exception:
            continue
    
    # Compute summary statistics
    depths = [r["depth_m"] for r in results.values() if "depth_m" in r]
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "station_count": len(results),
        "stations": results,
        "summary": {
            "depth_min_m": min(depths) if depths else None,
            "depth_max_m": max(depths) if depths else None,
            "depth_mean_m": round(sum(depths) / len(depths), 1) if depths else None,
            "depth_categories": _count_categories(results),
        }
    }


@router.get("/bathymetry/{station_id}")
async def get_station_bathymetry(station_id: str) -> Dict[str, Any]:
    """Get bathymetry analysis for a specific station."""
    if station_id not in ALL_STATIONS:
        raise HTTPException(status_code=404, detail=f"Station '{station_id}' not found")
    
    analyzer = BathymetryAnalyzer()
    
    try:
        analysis = await analyzer.analyze_station_bathymetry(station_id)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
def _compute_direction(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """Compute compass direction from point 1 to point 2."""
    import math
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    angle = math.degrees(math.atan2(dlon, dlat))
    angle = (angle + 360) % 360
    
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = int((angle + 22.5) / 45) % 8
    
    return directions[index]


def _to_cardinal(direction: str) -> str:
    """Convert 8-point direction to 4-point cardinal."""
    mapping = {
        "N": "N", "NE": "E", "E": "E", "SE": "S",
        "S": "S", "SW": "W", "W": "W", "NW": "N"
    }
    return mapping.get(direction, direction)


def _estimate_flow_speed(patterns: List[Dict]) -> Optional[float]:
    """Estimate weather flow speed from lag patterns."""
    if not patterns:
        return None
    
    from geopy.distance import geodesic
    
    speeds = []
    for p in patterns[:5]:
        if p["lag_hours"] > 0:
            source = ALL_STATIONS.get(p["source_station"])
            target = ALL_STATIONS.get(p["target_station"])
            
            if source and target:
                distance = geodesic(
                    (source["lat"], source["lon"]),
                    (target["lat"], target["lon"])
                ).kilometers
                
                speed = distance / p["lag_hours"]
                speeds.append(speed)
    
    return round(sum(speeds) / len(speeds), 1) if speeds else None


def _count_categories(results: Dict) -> Dict[str, int]:
    """Count stations by depth category."""
    categories = {}
    for r in results.values():
        cat = r.get("depth_category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    return categories


# Need asyncio for the cluster analysis fallback
import asyncio
