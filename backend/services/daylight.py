"""Daylight/sunset calculations using sunrise-sunset.org API."""

from datetime import datetime, date, time, timedelta
import httpx

from backend.models.trail import Coordinate


SUNRISE_SUNSET_API = "https://api.sunrise-sunset.org/json"


async def get_daylight_info(
    lat: float,
    lon: float,
    target_date: date | None = None
) -> dict:
    """Get sunrise/sunset times for a location and date.

    Returns:
        dict with keys: sunrise, sunset, daylight_hours, solar_noon
        Times are in local timezone (Mountain Time for Boise)
    """
    target_date = target_date or date.today()

    params = {
        "lat": lat,
        "lng": lon,
        "date": target_date.isoformat(),
        "formatted": 0,  # Get ISO 8601 format
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(SUNRISE_SUNSET_API, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()

        if data["status"] != "OK":
            raise ValueError(f"API error: {data.get('status')}")

        results = data["results"]

        # Parse times (they come in UTC)
        sunrise_utc = datetime.fromisoformat(results["sunrise"].replace("Z", "+00:00"))
        sunset_utc = datetime.fromisoformat(results["sunset"].replace("Z", "+00:00"))
        solar_noon_utc = datetime.fromisoformat(results["solar_noon"].replace("Z", "+00:00"))

        # Convert to Mountain Time (Boise is UTC-7 in summer, UTC-8 in winter)
        # Simple approach: check if date is in DST range
        is_dst = _is_dst(target_date)
        offset = timedelta(hours=-6 if is_dst else -7)

        sunrise_local = sunrise_utc + offset
        sunset_local = sunset_utc + offset
        solar_noon_local = solar_noon_utc + offset

        daylight_seconds = int(results["day_length"])
        daylight_hours = daylight_seconds / 3600

        return {
            "date": target_date.isoformat(),
            "sunrise": sunrise_local.strftime("%H:%M"),
            "sunset": sunset_local.strftime("%H:%M"),
            "solar_noon": solar_noon_local.strftime("%H:%M"),
            "daylight_hours": round(daylight_hours, 2),
            "sunrise_datetime": sunrise_local,
            "sunset_datetime": sunset_local,
        }

    except Exception as e:
        # Fallback: estimate for Boise area
        return _estimate_daylight(lat, lon, target_date)


def _is_dst(d: date) -> bool:
    """Check if date is in Daylight Saving Time (US rules)."""
    # DST starts second Sunday of March, ends first Sunday of November
    year = d.year

    # Find second Sunday of March
    march_1 = date(year, 3, 1)
    days_until_sunday = (6 - march_1.weekday()) % 7
    first_sunday_march = march_1 + timedelta(days=days_until_sunday)
    dst_start = first_sunday_march + timedelta(days=7)

    # Find first Sunday of November
    nov_1 = date(year, 11, 1)
    days_until_sunday = (6 - nov_1.weekday()) % 7
    dst_end = nov_1 + timedelta(days=days_until_sunday)

    return dst_start <= d < dst_end


def _estimate_daylight(lat: float, lon: float, target_date: date) -> dict:
    """Estimate daylight hours for Boise area based on date."""
    # Boise daylight varies from ~9 hours (Dec 21) to ~15.5 hours (Jun 21)
    day_of_year = target_date.timetuple().tm_yday

    # Approximate daylight hours using sine wave
    import math
    # Peak around day 172 (June 21)
    daylight_hours = 12.25 + 3.25 * math.sin(2 * math.pi * (day_of_year - 80) / 365)

    # Estimate sunrise/sunset
    solar_noon = time(13, 0)  # Approximate for Boise
    half_daylight = timedelta(hours=daylight_hours / 2)

    noon_dt = datetime.combine(target_date, solar_noon)
    sunrise_dt = noon_dt - half_daylight
    sunset_dt = noon_dt + half_daylight

    return {
        "date": target_date.isoformat(),
        "sunrise": sunrise_dt.strftime("%H:%M"),
        "sunset": sunset_dt.strftime("%H:%M"),
        "solar_noon": "13:00",
        "daylight_hours": round(daylight_hours, 2),
        "sunrise_datetime": sunrise_dt,
        "sunset_datetime": sunset_dt,
        "estimated": True,
    }


def get_daylight_sync(lat: float, lon: float, target_date: date | None = None) -> dict:
    """Synchronous version using estimation."""
    target_date = target_date or date.today()
    return _estimate_daylight(lat, lon, target_date)


def calculate_finish_time(
    start_time: datetime,
    hiking_hours: float,
    sunset_datetime: datetime,
    buffer_minutes: int = 30
) -> dict:
    """Calculate if a hike can be completed before dark.

    Args:
        start_time: When the hike starts
        hiking_hours: Estimated hiking duration
        sunset_datetime: When the sun sets
        buffer_minutes: Safety buffer before sunset

    Returns:
        dict with can_complete, finish_time, minutes_before_sunset
    """
    finish_time = start_time + timedelta(hours=hiking_hours)
    deadline = sunset_datetime - timedelta(minutes=buffer_minutes)
    minutes_before_sunset = (sunset_datetime - finish_time).total_seconds() / 60

    return {
        "can_complete": finish_time <= deadline,
        "finish_time": finish_time.strftime("%H:%M"),
        "minutes_before_sunset": round(minutes_before_sunset),
        "sunset": sunset_datetime.strftime("%H:%M"),
    }
