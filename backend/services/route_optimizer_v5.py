"""Route optimization v5 - Greedy loop-building approach.

Instead of solving globally and splitting, build efficient day-hikes one at a time:
1. Find good parking spots (trailheads)
2. From each parking spot, build the best possible loop hike
3. Repeat until all segments are covered

Key insight: 43% of segments are dead-ends that MUST be out-and-back.
The best we can do is ~69% efficiency (74mi mandatory garbage on 164.7mi challenge).
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from typing import Optional

import networkx as nx

from backend.models.trail import Coordinate, TrailSegment
from backend.services.graph_builder import TrailGraph, haversine_distance


class HikeCategory(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


@dataclass
class HikeSegment:
    seg_id: int
    seg_name: str
    length_mi: float
    is_challenge: bool


@dataclass
class PlannedHike:
    hike_id: int
    category: HikeCategory
    area_name: str
    parking_location: str
    parking_lat: float
    parking_lon: float

    segments: list[HikeSegment] = field(default_factory=list)
    segment_ids: set[int] = field(default_factory=set)

    challenge_miles: float = 0.0
    garbage_miles: float = 0.0
    estimated_hours: float = 0.0
    hiking_pace_mph: float = 2.5

    @property
    def total_miles(self) -> float:
        return self.challenge_miles + self.garbage_miles

    @property
    def challenge_hours(self) -> float:
        return self.challenge_miles / self.hiking_pace_mph

    @property
    def garbage_hours(self) -> float:
        return self.garbage_miles / self.hiking_pace_mph

    @property
    def efficiency(self) -> float:
        if self.total_miles == 0:
            return 0
        return (self.challenge_miles / self.total_miles) * 100


@dataclass
class OptimizationResult:
    hikes: list[PlannedHike]
    total_challenge_miles: float = 0.0
    total_garbage_miles: float = 0.0
    total_hikes: int = 0
    short_hikes: int = 0
    medium_hikes: int = 0
    long_hikes: int = 0

    @property
    def total_miles(self) -> float:
        return self.total_challenge_miles + self.total_garbage_miles

    @property
    def overall_efficiency(self) -> float:
        if self.total_miles == 0:
            return 0
        return (self.total_challenge_miles / self.total_miles) * 100


class LoopOptimizer:
    """Build efficient loop hikes from trailheads."""

    MAX_HIKE_HOURS = 6.0
    MAX_HIKE_MILES = 15.0  # Cap total miles per hike
    METERS_PER_MILE = 1609.34

    def __init__(
        self,
        trail_graph: TrailGraph,
        home_lat: float,
        home_lon: float,
        hiking_pace_mph: float = 2.5
    ):
        self.graph = trail_graph
        self.home_lat = home_lat
        self.home_lon = home_lon
        self.home_coord = Coordinate(lat=home_lat, lon=home_lon)
        self.hiking_pace_mph = hiking_pace_mph

        self._segments = {s.seg_id: s for s in self.graph.trail_data.segments}

        # Build multigraph
        self._trail_graph = self._build_graph()

        # Find trailheads (degree-1 nodes or well-connected junctions)
        self._trailheads = self._identify_trailheads()

    def _build_graph(self) -> nx.MultiGraph:
        g = nx.MultiGraph()

        for seg in self.graph.trail_data.segments:
            start = f"{seg.start_point.lat:.5f},{seg.start_point.lon:.5f}"
            end = f"{seg.end_point.lat:.5f},{seg.end_point.lon:.5f}"

            if start not in g:
                g.add_node(start, lat=seg.start_point.lat, lon=seg.start_point.lon)
            if end not in g:
                g.add_node(end, lat=seg.end_point.lat, lon=seg.end_point.lon)

            g.add_edge(start, end,
                       seg_id=seg.seg_id,
                       seg_name=seg.seg_name,
                       length_mi=seg.length_mi,
                       key=seg.seg_id)

        return g

    def _identify_trailheads(self) -> list[str]:
        """Find good parking/starting locations."""
        trailheads = []

        for node in self._trail_graph.nodes():
            degree = self._trail_graph.degree(node)
            # Good trailheads: degree 1 (dead-ends, common parking), degree 3+ (junctions)
            if degree == 1 or degree >= 3:
                trailheads.append(node)

        # Sort by distance from home
        trailheads.sort(key=lambda n: haversine_distance(
            self.home_coord,
            Coordinate(lat=self._trail_graph.nodes[n]['lat'],
                      lon=self._trail_graph.nodes[n]['lon'])
        ))

        return trailheads

    def _get_area_name(self, segment_ids: set[int]) -> str:
        names = set()
        for seg_id in segment_ids:
            seg = self._segments.get(seg_id)
            if seg:
                parts = seg.seg_name.rsplit(' ', 1)
                base = parts[0] if len(parts) > 1 and parts[1].isdigit() else seg.seg_name
                names.add(base)
        return " / ".join(sorted(names)[:3]) if names else "Trail Area"

    def _categorize(self, hours: float) -> HikeCategory:
        if hours <= 2:
            return HikeCategory.SHORT
        elif hours <= 4:
            return HikeCategory.MEDIUM
        return HikeCategory.LONG

    def _build_hike_from_node(
        self,
        start_node: str,
        remaining_segs: set[int],
        max_challenge_mi: float = 8.0
    ) -> Optional[PlannedHike]:
        """Build best possible hike from a starting node."""

        if not remaining_segs:
            return None

        # Use BFS to find reachable segments, prioritizing:
        # 1. Segments that form loops back to start
        # 2. Nearby uncompleted segments
        # 3. Dead-ends (do them in pairs when possible)

        hike_segs = []  # (seg_id, seg_name, length, is_challenge)
        hike_seg_ids = set()
        challenge_mi = 0
        garbage_mi = 0

        visited_nodes = {start_node}
        current_node = start_node

        def get_uncompleted_edges(node):
            """Get edges from node that have uncompleted challenge segments."""
            edges = []
            for neighbor in self._trail_graph.neighbors(node):
                for key, data in self._trail_graph.get_edge_data(node, neighbor).items():
                    seg_id = data['seg_id']
                    if seg_id in remaining_segs:
                        edges.append((neighbor, data))
            return edges

        def get_any_edges(node):
            """Get all edges from node."""
            edges = []
            for neighbor in self._trail_graph.neighbors(node):
                for key, data in self._trail_graph.get_edge_data(node, neighbor).items():
                    edges.append((neighbor, data))
            return edges

        # Phase 1: Greedy exploration of uncompleted segments
        while challenge_mi < max_challenge_mi:
            # Get uncompleted segments from current position
            uncompleted = get_uncompleted_edges(current_node)

            if not uncompleted:
                # No uncompleted segments reachable - try to find path to more
                # Find nearest node with uncompleted segments
                best_path = None
                best_dist = float('inf')

                for seg_id in remaining_segs:
                    seg = self._segments[seg_id]
                    for endpoint in [f"{seg.start_point.lat:.5f},{seg.start_point.lon:.5f}",
                                    f"{seg.end_point.lat:.5f},{seg.end_point.lon:.5f}"]:
                        if endpoint in self._trail_graph:
                            try:
                                path = nx.shortest_path(self._trail_graph, current_node, endpoint, weight='length_mi')
                                dist = nx.shortest_path_length(self._trail_graph, current_node, endpoint, weight='length_mi')
                                if dist < best_dist and dist < 3.0:  # Don't travel more than 3mi to next area
                                    best_dist = dist
                                    best_path = path
                            except nx.NetworkXNoPath:
                                pass

                if best_path and len(best_path) > 1:
                    # Walk the connecting path
                    for i in range(len(best_path) - 1):
                        u, v = best_path[i], best_path[i + 1]
                        edge_data = self._trail_graph.get_edge_data(u, v)
                        if edge_data:
                            shortest = min(edge_data.values(), key=lambda x: x['length_mi'])
                            seg_id = shortest['seg_id']

                            is_challenge = seg_id in remaining_segs and seg_id not in hike_seg_ids

                            hike_segs.append((seg_id, shortest['seg_name'], shortest['length_mi'], is_challenge))
                            if is_challenge:
                                challenge_mi += shortest['length_mi']
                                hike_seg_ids.add(seg_id)
                            else:
                                garbage_mi += shortest['length_mi']

                    current_node = best_path[-1]
                    visited_nodes.add(current_node)
                else:
                    break  # Can't reach more uncompleted segments

            else:
                # Pick best uncompleted segment
                # Prefer: shorter segments, segments that lead to more uncompleted
                def score_edge(edge):
                    neighbor, data = edge
                    seg_mi = data['length_mi']
                    # Check how many more uncompleted segments we can reach from neighbor
                    future = len(get_uncompleted_edges(neighbor))
                    # Lower score = better
                    return seg_mi - future * 0.5

                uncompleted.sort(key=score_edge)
                neighbor, data = uncompleted[0]
                seg_id = data['seg_id']

                hike_segs.append((seg_id, data['seg_name'], data['length_mi'], True))
                challenge_mi += data['length_mi']
                hike_seg_ids.add(seg_id)

                current_node = neighbor
                visited_nodes.add(current_node)

        # Phase 2: Return to start (or nearby trailhead)
        if current_node != start_node:
            try:
                return_path = nx.shortest_path(self._trail_graph, current_node, start_node, weight='length_mi')

                for i in range(len(return_path) - 1):
                    u, v = return_path[i], return_path[i + 1]
                    edge_data = self._trail_graph.get_edge_data(u, v)
                    if edge_data:
                        shortest = min(edge_data.values(), key=lambda x: x['length_mi'])
                        seg_id = shortest['seg_id']

                        is_challenge = seg_id in remaining_segs and seg_id not in hike_seg_ids

                        hike_segs.append((seg_id, shortest['seg_name'], shortest['length_mi'], is_challenge))
                        if is_challenge:
                            challenge_mi += shortest['length_mi']
                            hike_seg_ids.add(seg_id)
                        else:
                            garbage_mi += shortest['length_mi']

            except nx.NetworkXNoPath:
                # Can't return to start - this is an out-and-back
                # Backtrack along the path we came
                for seg_id, name, length, _ in reversed(hike_segs):
                    is_challenge = seg_id in remaining_segs and seg_id not in hike_seg_ids
                    hike_segs.append((seg_id, name + " (return)", length, is_challenge))
                    if is_challenge:
                        challenge_mi += length
                        hike_seg_ids.add(seg_id)
                    else:
                        garbage_mi += length

        if not hike_seg_ids:
            return None

        # Build hike object
        segments = [HikeSegment(seg_id=s[0], seg_name=s[1], length_mi=round(s[2], 2), is_challenge=s[3])
                   for s in hike_segs]

        start_data = self._trail_graph.nodes[start_node]
        total_mi = challenge_mi + garbage_mi
        hours = total_mi / self.hiking_pace_mph

        return PlannedHike(
            hike_id=0,
            category=self._categorize(hours),
            area_name=self._get_area_name(hike_seg_ids),
            parking_location=f"Trailhead ({start_data['lat']:.4f}, {start_data['lon']:.4f})",
            parking_lat=start_data['lat'],
            parking_lon=start_data['lon'],
            segments=segments,
            segment_ids=hike_seg_ids,
            challenge_miles=round(challenge_mi, 2),
            garbage_miles=round(garbage_mi, 2),
            estimated_hours=round(hours, 1),
            hiking_pace_mph=self.hiking_pace_mph,
        )

    def optimize(self, completed_segments: set[int] | None = None) -> OptimizationResult:
        """Build hikes one at a time from good starting points."""
        completed = completed_segments or set()
        remaining = set(self._segments.keys()) - completed

        if not remaining:
            return OptimizationResult(hikes=[])

        hikes = []

        # Target challenge miles per hike based on 6hr limit
        # With 50-70% efficiency, 8-10 challenge miles should fit in 6 hours
        target_challenge_mi = 8.0

        while remaining:
            # Find best starting point for next hike
            # Prefer trailheads close to remaining segments
            best_start = None
            best_score = float('inf')

            for node in self._trailheads:
                # Count nearby remaining segments
                nearby_remaining = 0
                for seg_id in remaining:
                    seg = self._segments[seg_id]
                    seg_start = f"{seg.start_point.lat:.5f},{seg.start_point.lon:.5f}"
                    seg_end = f"{seg.end_point.lat:.5f},{seg.end_point.lon:.5f}"
                    if seg_start == node or seg_end == node:
                        nearby_remaining += 1

                if nearby_remaining > 0:
                    # Score = distance from home - bonus for nearby segments
                    node_data = self._trail_graph.nodes[node]
                    dist_from_home = haversine_distance(
                        self.home_coord,
                        Coordinate(lat=node_data['lat'], lon=node_data['lon'])
                    ) / self.METERS_PER_MILE
                    score = dist_from_home - nearby_remaining * 2
                    if score < best_score:
                        best_score = score
                        best_start = node

            if not best_start:
                # No good trailhead found - use any node with remaining segments
                for seg_id in remaining:
                    seg = self._segments[seg_id]
                    best_start = f"{seg.start_point.lat:.5f},{seg.start_point.lon:.5f}"
                    break

            if not best_start:
                break

            # Build hike from this start
            hike = self._build_hike_from_node(best_start, remaining, target_challenge_mi)

            if not hike or not hike.segment_ids:
                # Remove this trailhead from consideration and try again
                if best_start in self._trailheads:
                    self._trailheads.remove(best_start)
                continue

            # Check time limit
            if hike.estimated_hours > self.MAX_HIKE_HOURS:
                # Hike too long - reduce target and try again
                target_challenge_mi = max(3, target_challenge_mi - 2)
                continue

            hikes.append(hike)
            remaining -= hike.segment_ids

            # Reset target for next hike
            target_challenge_mi = 8.0

        # Sort by distance from home
        hikes.sort(key=lambda h: haversine_distance(
            self.home_coord, Coordinate(lat=h.parking_lat, lon=h.parking_lon)
        ))

        # Renumber
        for i, h in enumerate(hikes):
            h.hike_id = i + 1

        total_challenge = sum(h.challenge_miles for h in hikes)
        total_garbage = sum(h.garbage_miles for h in hikes)

        return OptimizationResult(
            hikes=hikes,
            total_challenge_miles=round(total_challenge, 2),
            total_garbage_miles=round(total_garbage, 2),
            total_hikes=len(hikes),
            short_hikes=sum(1 for h in hikes if h.category == HikeCategory.SHORT),
            medium_hikes=sum(1 for h in hikes if h.category == HikeCategory.MEDIUM),
            long_hikes=sum(1 for h in hikes if h.category == HikeCategory.LONG),
        )

    def get_hike_for_time(
        self,
        available_hours: float,
        completed_segments: set[int] | None = None
    ) -> PlannedHike | None:
        """Get a single hike that fits in available time."""
        completed = completed_segments or set()
        remaining = set(self._segments.keys()) - completed

        if not remaining:
            return None

        max_mi = available_hours * self.hiking_pace_mph * 0.6  # Leave room for garbage

        # Find closest trailhead with remaining segments
        for node in self._trailheads:
            has_remaining = False
            for seg_id in remaining:
                seg = self._segments[seg_id]
                if (f"{seg.start_point.lat:.5f},{seg.start_point.lon:.5f}" == node or
                    f"{seg.end_point.lat:.5f},{seg.end_point.lon:.5f}" == node):
                    has_remaining = True
                    break

            if has_remaining:
                hike = self._build_hike_from_node(node, remaining, max_mi)
                if hike and hike.estimated_hours <= available_hours:
                    hike.hike_id = 1
                    return hike

        return None


def create_optimizer_v5(
    trail_graph: TrailGraph,
    home_lat: float,
    home_lon: float,
    hiking_pace_mph: float = 2.5
) -> LoopOptimizer:
    return LoopOptimizer(trail_graph, home_lat, home_lon, hiking_pace_mph)
