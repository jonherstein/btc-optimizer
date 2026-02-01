import json
import httpx
from pathlib import Path

from backend.config import settings
from backend.models.trail import TrailData


def load_trail_data_from_cache() -> TrailData:
    """Load trail data from local cache file."""
    cache_path = settings.trail_cache_file
    if not cache_path.exists():
        raise FileNotFoundError(f"Trail cache not found at {cache_path}")

    with open(cache_path) as f:
        data = json.load(f)

    return TrailData.from_api(data)


async def fetch_trail_data_from_api() -> TrailData:
    """Fetch trail data from BTC API and cache it."""
    url = settings.btc_trail_api
    params = {
        "code": settings.btc_trail_api_code,
        "MgrId": settings.btc_mgr_id,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()

    # Cache the data
    settings.trail_cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.trail_cache_file, "w") as f:
        json.dump(data, f)

    return TrailData.from_api(data)


def get_trail_data() -> TrailData:
    """Get trail data from cache (sync version for startup)."""
    return load_trail_data_from_cache()
