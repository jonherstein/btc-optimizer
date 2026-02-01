from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Paths
    data_dir: Path = Path(__file__).parent.parent / "data"
    trail_cache_file: Path = data_dir / "trail_cache.json"

    # User settings (can be overridden via environment or settings file)
    home_address: str = "3410 E Vortex Drive, Boise, ID 83712"
    home_lat: float = 43.5835  # Boulder Heights Estates area
    home_lon: float = -116.1428
    hiking_pace_mph: float = 2.5

    # API keys
    openroute_api_key: str = ""

    # BTC API (codes from the website - set via .env file)
    btc_trail_api: str = "https://btc-api.azurewebsites.net/api/GETChallengeTrailData_v2"
    btc_trail_api_code: str = ""  # Required - get from BTC website
    btc_mgr_id: int = 1

    # BTC Progress API (for tracking completed segments - requires active challenge)
    btc_progress_api: str = "https://btc-api.azurewebsites.net/api/GETAthleteActivities_v2"
    btc_progress_api_code: str = ""
    btc_user_id: str = ""  # User's BTC UID from the website

    # Track last sync time
    btc_last_sync: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
