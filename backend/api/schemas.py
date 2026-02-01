"""Pydantic schemas for API request/response."""

from pydantic import BaseModel
from datetime import date, datetime


class SettingsUpdate(BaseModel):
    home_address: str | None = None
    home_lat: float | None = None
    home_lon: float | None = None
    hiking_pace_mph: float | None = None


class SettingsResponse(BaseModel):
    home_address: str
    home_lat: float
    home_lon: float
    hiking_pace_mph: float


class BTCSettingsUpdate(BaseModel):
    btc_trail_api: str | None = None
    btc_trail_api_code: str | None = None
    btc_mgr_id: int | None = None
    btc_progress_api: str | None = None
    btc_progress_api_code: str | None = None
    btc_user_id: str | None = None


class BTCSettingsResponse(BaseModel):
    btc_trail_api: str
    btc_trail_api_code: str
    btc_mgr_id: int
    btc_progress_api: str
    btc_progress_api_code: str
    btc_user_id: str
    last_sync: str | None = None


class TimeBudgetRequest(BaseModel):
    available_hours: float
    target_date: date | None = None
    start_time: str | None = None  # HH:MM format
    prefer_close: bool = True


class SegmentInfo(BaseModel):
    seg_id: int
    seg_name: str
    length_mi: float
    direction: str


class TripResponse(BaseModel):
    trip_id: int
    segments: list[SegmentInfo]
    # Miles breakdown
    total_challenge_miles: float
    total_garbage_miles: float  # return/connectors
    total_miles: float
    # Time breakdown
    challenge_hours: float
    garbage_hours: float
    total_hours: float
    # Driving
    drive_miles: float | None = None
    drive_minutes: float | None = None
    # Location - Start
    center_lat: float
    center_lon: float
    trailhead_lat: float | None = None
    trailhead_lon: float | None = None
    parking_location: str | None = None
    # Location - End (for shuttle hikes)
    end_lat: float | None = None
    end_lon: float | None = None
    end_location: str | None = None
    # Shuttle info
    needs_shuttle: bool = False
    shuttle_savings_miles: float = 0.0
    shuttle_note: str | None = None
    # Metadata
    area_name: str | None = None
    category: str | None = None  # short, medium, long
    efficiency: float | None = None  # % of miles that are challenge


class OptimizedRouteResponse(BaseModel):
    trips: list[TripResponse]
    total_challenge_miles: float
    total_connector_miles: float
    total_trips: int
    shuttle_trips: int = 0
    total_shuttle_savings: float = 0.0
    efficiency: float = 0.0


class TimeBudgetResponse(BaseModel):
    trip: TripResponse | None
    daylight_info: dict
    drive_time_minutes: float
    can_complete_before_dark: bool
    message: str


class TrailStatsResponse(BaseModel):
    total_segments: int
    total_miles: float
    num_trailheads: int
    directional_breakdown: dict


class GraphDataResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]
    segments: list[dict]
