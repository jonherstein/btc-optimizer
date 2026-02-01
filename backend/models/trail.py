from pydantic import BaseModel
from typing import Literal
from enum import Enum


class Direction(str, Enum):
    BOTH = "both"
    UP = "up"
    DOWN = "down"


class Coordinate(BaseModel):
    lon: float
    lat: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.lon, self.lat)


class MasterTrail(BaseModel):
    master_trail_id: int
    master_trail_name: str
    bturl: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @classmethod
    def from_api(cls, data: dict) -> "MasterTrail":
        return cls(
            master_trail_id=data["masterTrailId"],
            master_trail_name=data["masterTrailName"],
            bturl=data["bturl"],
            min_x=data["minX"],
            min_y=data["minY"],
            max_x=data["maxX"],
            max_y=data["maxY"],
        )


class TrailSegment(BaseModel):
    seg_id: int
    seg_name: str
    length_ft: float
    length_mi: float
    direction: Direction
    special_instructions: str
    coordinates: list[Coordinate]

    @property
    def start_point(self) -> Coordinate:
        return self.coordinates[0]

    @property
    def end_point(self) -> Coordinate:
        return self.coordinates[-1]

    @classmethod
    def from_api(cls, feature: dict) -> "TrailSegment":
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]

        direction_map = {
            "both": Direction.BOTH,
            "up": Direction.UP,
            "down": Direction.DOWN,
        }

        return cls(
            seg_id=props["segId"],
            seg_name=props["segName"],
            length_ft=props["LengthFt"],
            length_mi=props["LengthFt"] / 5280,
            direction=direction_map.get(props["direction"], Direction.BOTH),
            special_instructions=props.get("specInst", ""),
            coordinates=[Coordinate(lon=c[0], lat=c[1]) for c in coords],
        )


class TrailData(BaseModel):
    last_updated_utc: str
    master_trails: list[MasterTrail]
    segments: list[TrailSegment]

    @classmethod
    def from_api(cls, data: dict) -> "TrailData":
        return cls(
            last_updated_utc=data["lastUpdatedUTC"],
            master_trails=[MasterTrail.from_api(t) for t in data["masterTrails"]],
            segments=[TrailSegment.from_api(s) for s in data["trailSegments"]],
        )

    @property
    def total_challenge_miles(self) -> float:
        return sum(s.length_mi for s in self.segments)
