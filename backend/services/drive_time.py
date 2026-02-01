"""Drive time calculations using OpenRouteService API."""

import httpx
from backend.config import settings
from backend.models.trail import Coordinate


OPENROUTE_DIRECTIONS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"


async def get_drive_time(
    origin: Coordinate,
    destination: Coordinate,
    api_key: str | None = None
) -> dict:
    """Get drive time and distance between two points.

    Returns:
        dict with keys: duration_seconds, duration_minutes, distance_meters, distance_miles
    """
    api_key = api_key or settings.openroute_api_key

    if not api_key:
        # Fallback: estimate based on straight-line distance
        # Assume average speed of 30 mph in Boise area (mix of city/highway)
        from backend.services.graph_builder import haversine_distance
        dist_m = haversine_distance(origin, destination)
        dist_mi = dist_m / 1609.34
        # Assume road distance is 1.3x straight-line (typical for road networks)
        road_dist_mi = dist_mi * 1.3
        duration_min = (road_dist_mi / 30) * 60  # 30 mph average

        return {
            "duration_seconds": duration_min * 60,
            "duration_minutes": duration_min,
            "distance_meters": dist_m * 1.3,
            "distance_miles": road_dist_mi,
            "estimated": True,
        }

    # Use OpenRouteService API
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }
    body = {
        "coordinates": [
            [origin.lon, origin.lat],
            [destination.lon, destination.lat]
        ],
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            OPENROUTE_DIRECTIONS_URL,
            headers=headers,
            json=body,
            timeout=10.0
        )
        response.raise_for_status()
        data = response.json()

    route = data["routes"][0]["summary"]
    return {
        "duration_seconds": route["duration"],
        "duration_minutes": route["duration"] / 60,
        "distance_meters": route["distance"],
        "distance_miles": route["distance"] / 1609.34,
        "estimated": False,
    }


def estimate_drive_time_sync(origin: Coordinate, destination: Coordinate) -> dict:
    """Synchronous version using straight-line estimation."""
    from backend.services.graph_builder import haversine_distance

    dist_m = haversine_distance(origin, destination)
    dist_mi = dist_m / 1609.34
    road_dist_mi = dist_mi * 1.3
    duration_min = (road_dist_mi / 30) * 60

    return {
        "duration_seconds": duration_min * 60,
        "duration_minutes": duration_min,
        "distance_meters": dist_m * 1.3,
        "distance_miles": road_dist_mi,
        "estimated": True,
    }
