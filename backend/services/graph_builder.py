import math
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx

from backend.models.trail import TrailData, TrailSegment, Direction, Coordinate


# Approximate meters per degree at Boise's latitude (~43.6°N)
METERS_PER_DEG_LAT = 111_000
METERS_PER_DEG_LON = 81_000  # cos(43.6°) * 111,000

# Maximum distance (in meters) to consider two endpoints as connected
CONNECTION_THRESHOLD_M = 50


def haversine_distance(coord1: Coordinate, coord2: Coordinate) -> float:
    """Calculate distance between two coordinates in meters."""
    lat1, lon1 = math.radians(coord1.lat), math.radians(coord1.lon)
    lat2, lon2 = math.radians(coord2.lat), math.radians(coord2.lon)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return 6371000 * c  # Earth radius in meters


def coord_to_grid_key(coord: Coordinate, cell_size_m: float = 100) -> tuple[int, int]:
    """Convert coordinate to grid cell for spatial indexing."""
    lat_cell = int(coord.lat * METERS_PER_DEG_LAT / cell_size_m)
    lon_cell = int(coord.lon * METERS_PER_DEG_LON / cell_size_m)
    return (lat_cell, lon_cell)


@dataclass
class NodeInfo:
    """Information about a graph node (junction point)."""
    coord: Coordinate
    connected_segments: list[int]  # segment IDs
    is_trailhead: bool = False  # True if likely a trailhead (degree 1 or parking nearby)


class TrailGraph:
    """Graph representation of trail network for route optimization."""

    def __init__(self, trail_data: TrailData):
        self.trail_data = trail_data
        self.graph = nx.MultiDiGraph()  # Directed to handle up/down constraints
        self.nodes: dict[str, NodeInfo] = {}  # node_id -> NodeInfo
        self.segment_lookup: dict[int, TrailSegment] = {s.seg_id: s for s in trail_data.segments}

        self._build_graph()

    def _coord_to_node_id(self, coord: Coordinate) -> str:
        """Create a unique node ID from coordinate (rounded for matching)."""
        # Round to ~5 decimal places (~1 meter precision)
        return f"{coord.lat:.5f},{coord.lon:.5f}"

    def _find_nearby_nodes(self, coord: Coordinate, threshold_m: float = CONNECTION_THRESHOLD_M) -> list[str]:
        """Find existing nodes within threshold distance of a coordinate."""
        nearby = []
        for node_id, node_info in self.nodes.items():
            dist = haversine_distance(coord, node_info.coord)
            if dist <= threshold_m:
                nearby.append(node_id)
        return nearby

    def _get_or_create_node(self, coord: Coordinate) -> str:
        """Get existing node or create new one for a coordinate."""
        # First check for nearby existing nodes
        nearby = self._find_nearby_nodes(coord)
        if nearby:
            # Use the closest existing node
            closest = min(nearby, key=lambda n: haversine_distance(coord, self.nodes[n].coord))
            return closest

        # Create new node
        node_id = self._coord_to_node_id(coord)
        if node_id not in self.nodes:
            self.nodes[node_id] = NodeInfo(coord=coord, connected_segments=[])
            self.graph.add_node(node_id, lat=coord.lat, lon=coord.lon)
        return node_id

    def _build_graph(self):
        """Build the trail graph from segment data."""
        for segment in self.trail_data.segments:
            start_node = self._get_or_create_node(segment.start_point)
            end_node = self._get_or_create_node(segment.end_point)

            # Track which segments connect to each node
            self.nodes[start_node].connected_segments.append(segment.seg_id)
            self.nodes[end_node].connected_segments.append(segment.seg_id)

            # Add edges based on direction constraints
            edge_data = {
                "seg_id": segment.seg_id,
                "seg_name": segment.seg_name,
                "length_mi": segment.length_mi,
                "length_ft": segment.length_ft,
                "direction": segment.direction.value,
                "is_challenge": True,  # All segments from BTC are challenge segments
            }

            if segment.direction == Direction.BOTH:
                # Bidirectional - add edges in both directions
                self.graph.add_edge(start_node, end_node, **edge_data)
                self.graph.add_edge(end_node, start_node, **edge_data)
            elif segment.direction == Direction.UP:
                # Up only - typically from lower elevation to higher
                # Assume start -> end is the "up" direction
                self.graph.add_edge(start_node, end_node, **edge_data)
            elif segment.direction == Direction.DOWN:
                # Down only - from higher to lower elevation
                self.graph.add_edge(end_node, start_node, **edge_data)

        # Mark trailheads (nodes with low degree, likely access points)
        for node_id, node_info in self.nodes.items():
            in_deg = self.graph.in_degree(node_id)
            out_deg = self.graph.out_degree(node_id)
            # A trailhead typically has degree 1-2 (one trail going in/out)
            if in_deg + out_deg <= 2:
                node_info.is_trailhead = True

    def get_trailheads(self) -> list[str]:
        """Get all likely trailhead nodes."""
        return [nid for nid, info in self.nodes.items() if info.is_trailhead]

    def get_segment(self, seg_id: int) -> TrailSegment | None:
        """Get segment by ID."""
        return self.segment_lookup.get(seg_id)

    def get_all_segment_ids(self) -> list[int]:
        """Get all challenge segment IDs."""
        return list(self.segment_lookup.keys())

    def get_node_coord(self, node_id: str) -> Coordinate:
        """Get coordinate for a node."""
        return self.nodes[node_id].coord

    def find_nearest_node(self, lat: float, lon: float) -> str:
        """Find the nearest node to a given coordinate."""
        target = Coordinate(lat=lat, lon=lon)
        min_dist = float("inf")
        nearest = None

        for node_id, node_info in self.nodes.items():
            dist = haversine_distance(target, node_info.coord)
            if dist < min_dist:
                min_dist = dist
                nearest = node_id

        return nearest

    def get_stats(self) -> dict:
        """Get graph statistics."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_segments": len(self.segment_lookup),
            "num_trailheads": len(self.get_trailheads()),
            "total_challenge_miles": sum(s.length_mi for s in self.segment_lookup.values()),
            "directional_segments": {
                "both": sum(1 for s in self.segment_lookup.values() if s.direction == Direction.BOTH),
                "up": sum(1 for s in self.segment_lookup.values() if s.direction == Direction.UP),
                "down": sum(1 for s in self.segment_lookup.values() if s.direction == Direction.DOWN),
            },
        }


def build_trail_graph(trail_data: TrailData) -> TrailGraph:
    """Build trail graph from trail data."""
    return TrailGraph(trail_data)
