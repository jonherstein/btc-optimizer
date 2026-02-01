"""API routes for the Boise Trails Challenge optimizer."""

from datetime import date, datetime, timedelta
from fastapi import APIRouter

from backend.config import settings
from backend.models.trail import Coordinate
from backend.services.btc_client import get_trail_data
from backend.services.graph_builder import build_trail_graph, TrailGraph, haversine_distance
from backend.services.route_optimizer_v6 import create_optimizer_v6, PlannedHike
from backend.services.drive_time import estimate_drive_time_sync
from backend.services.daylight import get_daylight_sync, calculate_finish_time
from backend.api.schemas import (
    SettingsUpdate, SettingsResponse,
    BTCSettingsUpdate, BTCSettingsResponse,
    TimeBudgetRequest, TimeBudgetResponse,
    TripResponse, SegmentInfo, OptimizedRouteResponse,
    TrailStatsResponse, GraphDataResponse
)

router = APIRouter()

# Cache for trail data and graph (loaded once at startup)
_trail_data = None
_trail_graph: TrailGraph | None = None


def get_graph() -> TrailGraph:
    """Get or create the trail graph."""
    global _trail_data, _trail_graph
    if _trail_graph is None:
        _trail_data = get_trail_data()
        _trail_graph = build_trail_graph(_trail_data)
    return _trail_graph


def hike_to_response(hike: PlannedHike, home_coord: Coordinate | None = None) -> TripResponse:
    """Convert PlannedHike to API response."""
    segments = []
    for seg in hike.segments:
        segments.append(SegmentInfo(
            seg_id=seg.seg_id,
            seg_name=seg.seg_name,
            length_mi=seg.length_mi,
            direction="challenge" if seg.is_challenge else "garbage",
        ))

    # Calculate drive distance/time
    drive_miles = None
    drive_minutes = None
    if home_coord:
        hike_coord = Coordinate(lat=hike.start_lat, lon=hike.start_lon)
        # Straight-line distance * 1.4 road factor
        direct_m = haversine_distance(home_coord, hike_coord)
        drive_miles = round((direct_m / 1609.34) * 1.4, 1)
        # Estimate 30 mph average for mountain roads
        drive_minutes = round((drive_miles / 30) * 60, 0)

    return TripResponse(
        trip_id=hike.hike_id,
        segments=segments,
        # Miles breakdown
        total_challenge_miles=hike.challenge_miles,
        total_garbage_miles=hike.garbage_miles,
        total_miles=round(hike.total_miles, 2),
        # Time breakdown
        challenge_hours=round(hike.challenge_hours, 2),
        garbage_hours=round(hike.garbage_hours, 2),
        total_hours=hike.estimated_hours,
        # Driving
        drive_miles=drive_miles,
        drive_minutes=drive_minutes,
        # Location - Start
        center_lat=hike.start_lat,
        center_lon=hike.start_lon,
        trailhead_lat=hike.start_lat,
        trailhead_lon=hike.start_lon,
        parking_location=hike.start_location,
        # Location - End (for shuttle hikes)
        end_lat=hike.end_lat,
        end_lon=hike.end_lon,
        end_location=hike.end_location,
        # Shuttle info
        needs_shuttle=hike.needs_shuttle,
        shuttle_savings_miles=hike.shuttle_savings_miles,
        shuttle_note=hike.shuttle_note if hike.needs_shuttle else None,
        # Metadata
        area_name=hike.area_name,
        category=hike.category.value,
        efficiency=round(hike.efficiency, 1),
    )


@router.get("/trails/stats", response_model=TrailStatsResponse)
def get_trail_stats():
    """Get trail statistics."""
    graph = get_graph()
    stats = graph.get_stats()
    return TrailStatsResponse(
        total_segments=stats["num_segments"],
        total_miles=round(stats["total_challenge_miles"], 1),
        num_trailheads=stats["num_trailheads"],
        directional_breakdown=stats["directional_segments"],
    )


@router.get("/trails/graph", response_model=GraphDataResponse)
def get_trail_graph_data():
    """Get trail graph data for map visualization."""
    graph = get_graph()

    nodes = []
    for node_id, node_info in graph.nodes.items():
        nodes.append({
            "id": node_id,
            "lat": node_info.coord.lat,
            "lon": node_info.coord.lon,
            "is_trailhead": node_info.is_trailhead,
        })

    segments = []
    for seg in graph.trail_data.segments:
        segments.append({
            "seg_id": seg.seg_id,
            "seg_name": seg.seg_name,
            "length_mi": round(seg.length_mi, 2),
            "direction": seg.direction.value,
            "coordinates": [[c.lon, c.lat] for c in seg.coordinates],
        })

    return GraphDataResponse(nodes=nodes, edges=[], segments=segments)


@router.get("/settings", response_model=SettingsResponse)
def get_settings():
    """Get current user settings."""
    return SettingsResponse(
        home_address=settings.home_address,
        home_lat=settings.home_lat,
        home_lon=settings.home_lon,
        hiking_pace_mph=settings.hiking_pace_mph,
    )


@router.put("/settings", response_model=SettingsResponse)
def update_settings(update: SettingsUpdate):
    """Update user settings."""
    if update.home_address is not None:
        settings.home_address = update.home_address
    if update.home_lat is not None:
        settings.home_lat = update.home_lat
    if update.home_lon is not None:
        settings.home_lon = update.home_lon
    if update.hiking_pace_mph is not None:
        settings.hiking_pace_mph = update.hiking_pace_mph

    return get_settings()


@router.get("/btc-settings", response_model=BTCSettingsResponse)
def get_btc_settings():
    """Get BTC API configuration."""
    return BTCSettingsResponse(
        btc_trail_api=settings.btc_trail_api,
        btc_trail_api_code=settings.btc_trail_api_code,
        btc_mgr_id=settings.btc_mgr_id,
        btc_progress_api=settings.btc_progress_api,
        btc_progress_api_code=settings.btc_progress_api_code,
        btc_user_id=settings.btc_user_id,
        last_sync=settings.btc_last_sync or None,
    )


@router.put("/btc-settings", response_model=BTCSettingsResponse)
def update_btc_settings(update: BTCSettingsUpdate):
    """Update BTC API configuration."""
    if update.btc_trail_api is not None:
        settings.btc_trail_api = update.btc_trail_api
    if update.btc_trail_api_code is not None:
        settings.btc_trail_api_code = update.btc_trail_api_code
    if update.btc_mgr_id is not None:
        settings.btc_mgr_id = update.btc_mgr_id
    if update.btc_progress_api is not None:
        settings.btc_progress_api = update.btc_progress_api
    if update.btc_progress_api_code is not None:
        settings.btc_progress_api_code = update.btc_progress_api_code
    if update.btc_user_id is not None:
        settings.btc_user_id = update.btc_user_id

    return get_btc_settings()


@router.post("/btc-sync")
async def sync_trail_data():
    """Refresh trail data from BTC API."""
    from backend.services.btc_client import fetch_trail_data_from_api
    global _trail_data, _trail_graph

    try:
        _trail_data = await fetch_trail_data_from_api()
        _trail_graph = build_trail_graph(_trail_data)
        settings.btc_last_sync = datetime.now().isoformat()
        return {
            "success": True,
            "message": f"Synced {len(_trail_data.segments)} segments",
            "last_sync": settings.btc_last_sync,
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Sync failed: {str(e)}",
        }


@router.get("/optimize")
def optimize_full_challenge():
    """Get optimized route for the full challenge using shuttle-aware algorithm."""
    graph = get_graph()
    optimizer = create_optimizer_v6(
        graph,
        settings.home_lat,
        settings.home_lon,
        settings.hiking_pace_mph
    )

    result = optimizer.optimize()
    home_coord = Coordinate(lat=settings.home_lat, lon=settings.home_lon)

    return {
        "trips": [hike_to_response(h, home_coord) for h in result.hikes],
        "total_challenge_miles": result.total_challenge_miles,
        "total_garbage_miles": result.total_garbage_miles,
        "total_miles": result.total_miles,
        "total_trips": result.total_hikes,
        "efficiency": round(result.overall_efficiency, 1),
        "shuttle_trips": result.shuttle_hikes,
        "total_shuttle_savings": result.total_shuttle_savings,
        "by_category": {
            "short": result.short_hikes,
            "medium": result.medium_hikes,
            "long": result.long_hikes,
        }
    }


@router.post("/time-budget")
def get_time_budget_trip(request: TimeBudgetRequest):
    """Get recommended trip for available time budget."""
    graph = get_graph()
    optimizer = create_optimizer_v6(
        graph,
        settings.home_lat,
        settings.home_lon,
        settings.hiking_pace_mph
    )

    # Get daylight info
    target_date = request.target_date or date.today()
    daylight = get_daylight_sync(settings.home_lat, settings.home_lon, target_date)

    # Get recommended trip
    hike = optimizer.get_hike_for_time(request.available_hours)

    if not hike:
        return {
            "trip": None,
            "daylight_info": daylight,
            "drive_time_minutes": 0,
            "can_complete_before_dark": False,
            "message": "No suitable hikes found for the given time budget.",
        }

    # Estimate drive time
    hike_center = Coordinate(lat=hike.parking_lat, lon=hike.parking_lon)
    home = Coordinate(lat=settings.home_lat, lon=settings.home_lon)
    drive_info = estimate_drive_time_sync(home, hike_center)
    drive_time_min = drive_info["duration_minutes"]

    # Calculate if can complete before dark
    start_time_str = request.start_time or "09:00"
    start_hour, start_min = map(int, start_time_str.split(":"))
    start_datetime = datetime.combine(target_date, datetime.min.time().replace(hour=start_hour, minute=start_min))

    # Account for drive time to trailhead
    hike_start = start_datetime + timedelta(minutes=drive_time_min)
    finish_info = calculate_finish_time(
        hike_start,
        hike.estimated_hours,
        daylight["sunset_datetime"]
    )

    message = f"Recommended: {hike.area_name} ({hike.category.value})"
    message += f" - {hike.challenge_miles:.1f} challenge mi + {hike.garbage_miles:.1f} return/connector mi"
    message += f" = {hike.total_miles:.1f} total mi, ~{hike.estimated_hours:.1f} hrs"

    if finish_info["can_complete"]:
        message += f". Finish by {finish_info['finish_time']}, {finish_info['minutes_before_sunset']} min before sunset."
    else:
        message += f". Warning: May not complete before sunset ({daylight['sunset']})."

    return {
        "trip": hike_to_response(hike, home),
        "daylight_info": daylight,
        "drive_time_minutes": round(drive_time_min, 1),
        "can_complete_before_dark": finish_info["can_complete"],
        "message": message,
    }
