"""Route optimization for the Boise Trails Challenge.

This module provides algorithms to optimize the order of completing trail segments,
minimizing total distance (challenge + connector miles) while respecting constraints.
"""

from dataclasses import dataclass, field
from collections import defaultdict
import math

import networkx as nx

from backend.models.trail import Coordinate, Direction
from backend.services.graph_builder import TrailGraph, haversine_distance


@dataclass
class Trip:
    """A single hiking trip (one drive to a trailhead area)."""
    trip_id: int
    segments: list[int]  # Segment IDs in order
    trailhead_node: str  # Starting node
    total_challenge_miles: float = 0.0
    total_connector_miles: float = 0.0
    estimated_hiking_hours: float = 0.0
    center_lat: float = 0.0
    center_lon: float = 0.0

    @property
    def total_miles(self) -> float:
        return self.total_challenge_miles + self.total_connector_miles


@dataclass
class OptimizedRoute:
    """Complete optimized route for the challenge."""
    trips: list[Trip]
    total_challenge_miles: float
    total_connector_miles: float
    total_drive_time_hours: float = 0.0  # Filled in by drive time service

    @property
    def total_hiking_miles(self) -> float:
        return self.total_challenge_miles + self.total_connector_miles


class RouteOptimizer:
    """Optimizes routes for completing the Boise Trails Challenge."""

    def __init__(self, trail_graph: TrailGraph, home_lat: float, home_lon: float, hiking_pace_mph: float = 2.5):
        self.graph = trail_graph
        self.home_lat = home_lat
        self.home_lon = home_lon
        self.home_coord = Coordinate(lat=home_lat, lon=home_lon)
        self.hiking_pace_mph = hiking_pace_mph

    def _cluster_segments_geographically(self, segment_ids: list[int], max_cluster_radius_mi: float = 3.0) -> list[list[int]]:
        """Cluster segments by geographic proximity.

        Uses a simple greedy clustering approach - start with furthest segment from home,
        grow cluster until radius exceeded, repeat.
        """
        if not segment_ids:
            return []

        # Get centroid of each segment
        segment_centers: dict[int, Coordinate] = {}
        for seg_id in segment_ids:
            seg = self.graph.get_segment(seg_id)
            if seg:
                mid_idx = len(seg.coordinates) // 2
                segment_centers[seg_id] = seg.coordinates[mid_idx]

        remaining = set(segment_ids)
        clusters: list[list[int]] = []

        while remaining:
            # Find segment furthest from home among remaining
            furthest = max(
                remaining,
                key=lambda s: haversine_distance(self.home_coord, segment_centers[s])
            )

            # Start new cluster with this segment
            cluster = [furthest]
            remaining.remove(furthest)
            cluster_center = segment_centers[furthest]

            # Add nearby segments to cluster
            added = True
            while added and remaining:
                added = False
                for seg_id in list(remaining):
                    dist = haversine_distance(cluster_center, segment_centers[seg_id])
                    if dist <= max_cluster_radius_mi * 1609.34:  # Convert miles to meters
                        cluster.append(seg_id)
                        remaining.remove(seg_id)
                        # Update cluster center
                        lats = [segment_centers[s].lat for s in cluster]
                        lons = [segment_centers[s].lon for s in cluster]
                        cluster_center = Coordinate(
                            lat=sum(lats) / len(lats),
                            lon=sum(lons) / len(lons)
                        )
                        added = True

            clusters.append(cluster)

        return clusters

    def _order_segments_within_trip(self, segment_ids: list[int], start_node: str) -> list[int]:
        """Order segments within a trip to minimize backtracking.

        Uses a greedy nearest-neighbor approach starting from the trailhead.
        """
        if not segment_ids:
            return []

        if len(segment_ids) == 1:
            return segment_ids

        # Build a subgraph with only the segments we need to visit
        remaining = set(segment_ids)
        ordered = []
        current_node = start_node

        while remaining:
            # Find the nearest unvisited segment from current position
            best_seg = None
            best_dist = float("inf")
            best_target_node = None

            for seg_id in remaining:
                seg = self.graph.get_segment(seg_id)
                if not seg:
                    remaining.remove(seg_id)
                    continue

                # Find the nodes for this segment's endpoints
                start_node_id = self.graph._coord_to_node_id(seg.start_point)
                end_node_id = self.graph._coord_to_node_id(seg.end_point)

                # Try to find path to either endpoint
                for target_node in [start_node_id, end_node_id]:
                    try:
                        # Use shortest path length as distance metric
                        path_length = nx.shortest_path_length(
                            self.graph.graph, current_node, target_node, weight="length_mi"
                        )
                        if path_length < best_dist:
                            best_dist = path_length
                            best_seg = seg_id
                            # The other endpoint is where we'll end up
                            best_target_node = end_node_id if target_node == start_node_id else start_node_id
                    except nx.NetworkXNoPath:
                        # Try direct distance as fallback
                        target_coord = self.graph.get_node_coord(target_node) if target_node in self.graph.nodes else seg.start_point
                        current_coord = self.graph.get_node_coord(current_node)
                        dist_m = haversine_distance(current_coord, target_coord)
                        dist_mi = dist_m / 1609.34
                        if dist_mi < best_dist:
                            best_dist = dist_mi
                            best_seg = seg_id
                            best_target_node = end_node_id if target_node == start_node_id else start_node_id

            if best_seg is None:
                # Add remaining segments in any order
                ordered.extend(list(remaining))
                break

            ordered.append(best_seg)
            remaining.remove(best_seg)
            if best_target_node and best_target_node in self.graph.nodes:
                current_node = best_target_node

        return ordered

    def _find_best_trailhead_for_cluster(self, segment_ids: list[int]) -> str:
        """Find the best trailhead (entry point) for a cluster of segments."""
        # Get all nodes touched by these segments
        cluster_nodes = set()
        for seg_id in segment_ids:
            seg = self.graph.get_segment(seg_id)
            if seg:
                start_node = self.graph._coord_to_node_id(seg.start_point)
                end_node = self.graph._coord_to_node_id(seg.end_point)
                if start_node in self.graph.nodes:
                    cluster_nodes.add(start_node)
                if end_node in self.graph.nodes:
                    cluster_nodes.add(end_node)

        # Prefer actual trailheads
        trailheads = [n for n in cluster_nodes if self.graph.nodes[n].is_trailhead]

        if trailheads:
            # Return trailhead closest to home
            return min(
                trailheads,
                key=lambda n: haversine_distance(self.home_coord, self.graph.get_node_coord(n))
            )

        # Otherwise return any cluster node closest to home
        if cluster_nodes:
            return min(
                cluster_nodes,
                key=lambda n: haversine_distance(self.home_coord, self.graph.get_node_coord(n))
            )

        # Fallback to nearest node overall
        return self.graph.find_nearest_node(self.home_lat, self.home_lon)

    def _calculate_trip_stats(self, trip: Trip) -> Trip:
        """Calculate statistics for a trip."""
        challenge_miles = 0.0
        for seg_id in trip.segments:
            seg = self.graph.get_segment(seg_id)
            if seg:
                challenge_miles += seg.length_mi

        trip.total_challenge_miles = challenge_miles
        # TODO: Calculate connector miles by finding actual path through segments
        trip.total_connector_miles = 0.0  # Simplified for now
        trip.estimated_hiking_hours = trip.total_miles / self.hiking_pace_mph

        # Calculate center point
        lats, lons = [], []
        for seg_id in trip.segments:
            seg = self.graph.get_segment(seg_id)
            if seg:
                mid = seg.coordinates[len(seg.coordinates) // 2]
                lats.append(mid.lat)
                lons.append(mid.lon)
        if lats:
            trip.center_lat = sum(lats) / len(lats)
            trip.center_lon = sum(lons) / len(lons)

        return trip

    def _order_trips_by_distance_from_home(self, trips: list[Trip]) -> list[Trip]:
        """Order trips by distance from home (closest first) for efficiency."""
        return sorted(
            trips,
            key=lambda t: haversine_distance(
                self.home_coord,
                Coordinate(lat=t.center_lat, lon=t.center_lon)
            )
        )

    def optimize_full_challenge(
        self,
        completed_segment_ids: set[int] | None = None,
        max_trip_miles: float = 15.0,
        max_cluster_radius_mi: float = 3.0,
    ) -> OptimizedRoute:
        """Generate an optimized route for completing the full challenge.

        Args:
            completed_segment_ids: Set of already-completed segment IDs (excluded from route)
            max_trip_miles: Maximum miles per trip (for splitting large clusters)
            max_cluster_radius_mi: Maximum radius for geographic clustering

        Returns:
            OptimizedRoute with trips ordered for efficient completion
        """
        completed = completed_segment_ids or set()
        all_segments = self.graph.get_all_segment_ids()
        remaining_segments = [s for s in all_segments if s not in completed]

        if not remaining_segments:
            return OptimizedRoute(trips=[], total_challenge_miles=0, total_connector_miles=0)

        # Step 1: Cluster segments geographically
        clusters = self._cluster_segments_geographically(remaining_segments, max_cluster_radius_mi)

        # Step 2: Create trips from clusters
        trips = []
        for i, cluster in enumerate(clusters):
            trailhead = self._find_best_trailhead_for_cluster(cluster)
            ordered_segments = self._order_segments_within_trip(cluster, trailhead)

            trip = Trip(
                trip_id=i + 1,
                segments=ordered_segments,
                trailhead_node=trailhead,
            )
            trip = self._calculate_trip_stats(trip)

            # Split if too long
            if trip.total_challenge_miles > max_trip_miles and len(ordered_segments) > 1:
                # Simple split in half
                mid = len(ordered_segments) // 2
                trip1 = Trip(
                    trip_id=len(trips) + 1,
                    segments=ordered_segments[:mid],
                    trailhead_node=trailhead,
                )
                trip2 = Trip(
                    trip_id=len(trips) + 2,
                    segments=ordered_segments[mid:],
                    trailhead_node=trailhead,
                )
                trips.append(self._calculate_trip_stats(trip1))
                trips.append(self._calculate_trip_stats(trip2))
            else:
                trip.trip_id = len(trips) + 1
                trips.append(trip)

        # Step 3: Order trips by distance from home
        trips = self._order_trips_by_distance_from_home(trips)
        for i, trip in enumerate(trips):
            trip.trip_id = i + 1

        # Calculate totals
        total_challenge = sum(t.total_challenge_miles for t in trips)
        total_connector = sum(t.total_connector_miles for t in trips)

        return OptimizedRoute(
            trips=trips,
            total_challenge_miles=total_challenge,
            total_connector_miles=total_connector,
        )

    def get_trip_for_time_budget(
        self,
        available_hours: float,
        completed_segment_ids: set[int] | None = None,
        prefer_close: bool = True,
    ) -> Trip | None:
        """Get a recommended trip that fits within time budget.

        Args:
            available_hours: Total time available including hiking (not including drive)
            completed_segment_ids: Already completed segments
            prefer_close: If True, prefer trips closer to home

        Returns:
            A Trip that fits the time budget, or None if no suitable trip found
        """
        route = self.optimize_full_challenge(completed_segment_ids)

        if not route.trips:
            return None

        # Find trips that fit time budget
        suitable_trips = [
            t for t in route.trips
            if t.estimated_hiking_hours <= available_hours
        ]

        if not suitable_trips:
            # Return the shortest trip even if over budget
            return min(route.trips, key=lambda t: t.estimated_hiking_hours)

        if prefer_close:
            # Return closest suitable trip
            return min(
                suitable_trips,
                key=lambda t: haversine_distance(
                    self.home_coord,
                    Coordinate(lat=t.center_lat, lon=t.center_lon)
                )
            )
        else:
            # Return longest suitable trip (maximize progress)
            return max(suitable_trips, key=lambda t: t.total_challenge_miles)


def create_optimizer(trail_graph: TrailGraph, home_lat: float, home_lon: float, hiking_pace_mph: float = 2.5) -> RouteOptimizer:
    """Create a route optimizer instance."""
    return RouteOptimizer(trail_graph, home_lat, home_lon, hiking_pace_mph)
