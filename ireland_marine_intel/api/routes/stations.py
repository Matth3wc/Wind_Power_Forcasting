"""
Stations API routes.
"""
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from config.settings import ALL_STATIONS, BUOY_STATIONS, COASTAL_STATIONS, LIGHTHOUSE_STATIONS
from api.models.schemas import StationMetadata

router = APIRouter()


@router.get("", response_model=List[StationMetadata])
async def list_stations(
    station_type: Optional[str] = Query(
        None,
        description="Filter by station type: offshore_buoy, coastal_buoy, lighthouse, synoptic, observatory"
    )
) -> List[StationMetadata]:
    """
    List all monitoring stations.
    
    Returns metadata for all buoys, lighthouses, and coastal weather stations
    in the Irish marine monitoring network.
    """
    stations = []
    
    for station_id, info in ALL_STATIONS.items():
        # Apply type filter if specified
        if station_type and info.get("type") != station_type:
            continue
        
        stations.append(StationMetadata(
            station_id=station_id,
            name=info["name"],
            latitude=info["lat"],
            longitude=info["lon"],
            station_type=info["type"],
            depth_m=info.get("depth_m"),
            description=info.get("description")
        ))
    
    return stations


@router.get("/buoys", response_model=List[StationMetadata])
async def list_buoys() -> List[StationMetadata]:
    """List all buoy stations (offshore and coastal)."""
    stations = []
    
    for station_id, info in {**BUOY_STATIONS, **COASTAL_STATIONS}.items():
        stations.append(StationMetadata(
            station_id=station_id,
            name=info["name"],
            latitude=info["lat"],
            longitude=info["lon"],
            station_type=info["type"],
            depth_m=info.get("depth_m"),
            description=info.get("description")
        ))
    
    return stations


@router.get("/lighthouses", response_model=List[StationMetadata])
async def list_lighthouses() -> List[StationMetadata]:
    """List all lighthouse and coastal weather stations."""
    stations = []
    
    for station_id, info in LIGHTHOUSE_STATIONS.items():
        stations.append(StationMetadata(
            station_id=station_id,
            name=info["name"],
            latitude=info["lat"],
            longitude=info["lon"],
            station_type=info["type"],
            depth_m=info.get("depth_m"),
            description=info.get("description")
        ))
    
    return stations


@router.get("/{station_id}", response_model=StationMetadata)
async def get_station(station_id: str) -> StationMetadata:
    """
    Get details for a specific station.
    
    Args:
        station_id: Station identifier (e.g., 'M5', 'IL1', 'VALENTIA')
    """
    if station_id not in ALL_STATIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Station '{station_id}' not found"
        )
    
    info = ALL_STATIONS[station_id]
    
    return StationMetadata(
        station_id=station_id,
        name=info["name"],
        latitude=info["lat"],
        longitude=info["lon"],
        station_type=info["type"],
        depth_m=info.get("depth_m"),
        description=info.get("description")
    )


@router.get("/{station_id}/neighbors")
async def get_station_neighbors(
    station_id: str,
    max_distance_km: float = Query(200, description="Maximum distance in km")
) -> List[dict]:
    """
    Get neighboring stations within a specified distance.
    
    Returns list of stations sorted by distance from the specified station.
    """
    from geopy.distance import geodesic
    
    if station_id not in ALL_STATIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Station '{station_id}' not found"
        )
    
    source = ALL_STATIONS[station_id]
    source_coords = (source["lat"], source["lon"])
    
    neighbors = []
    
    for other_id, other_info in ALL_STATIONS.items():
        if other_id == station_id:
            continue
        
        other_coords = (other_info["lat"], other_info["lon"])
        distance = geodesic(source_coords, other_coords).kilometers
        
        if distance <= max_distance_km:
            neighbors.append({
                "station_id": other_id,
                "name": other_info["name"],
                "latitude": other_info["lat"],
                "longitude": other_info["lon"],
                "station_type": other_info["type"],
                "distance_km": round(distance, 1)
            })
    
    # Sort by distance
    neighbors.sort(key=lambda x: x["distance_km"])
    
    return neighbors
