"""Route optimization v4 - Chinese Postman Problem approach.

The Boise Trails Challenge is fundamentally the Chinese Postman Problem:
- We need to traverse every trail segment at least once
- We want to minimize total distance walked
- We can walk segments multiple times if needed

For optimal efficiency:
1. Find connected components in the trail graph
2. For each component, solve the Chinese Postman Problem
3. Split solutions into day-hike sized chunks (max 6 hours)
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from itertools import combinations

import networkx as nx

from backend.models.trail import Coordinate, TrailSegment
from backend.services.graph_builder import TrailGraph, haversine_distance


class HikeCategory(str, Enum):
    SHORT = "short"      # 1-2 hours
    MEDIUM = "medium"    # 2-4 hours
    LONG = "long"        # 4-6 hours


@dataclass
class HikeSegment:
    """A segment within a hike."""
    seg_id: int
    seg_name: str
    length_mi: float
    is_challenge: bool  # True = first time hiking this segment


@dataclass
class PlannedHike:
    """A complete hike plan."""
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
    """Complete challenge optimization result."""
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


class ChinesePostmanOptimizer:
    """Optimizes hiking routes using the Chinese Postman Problem solution."""

    MAX_HIKE_HOURS = 6.0
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

        # Build undirected multigraph for CPP
        self._trail_graph = self._build_trail_graph()
        self._segments = {s.seg_id: s for s in self.graph.trail_data.segments}

    def _build_trail_graph(self) -> nx.MultiGraph:
        """Build undirected multigraph from trail segments."""
        g = nx.MultiGraph()

        for seg in self.graph.trail_data.segments:
            # Node IDs from coordinates
            start = f"{seg.start_point.lat:.5f},{seg.start_point.lon:.5f}"
            end = f"{seg.end_point.lat:.5f},{seg.end_point.lon:.5f}"

            # Add nodes with coordinates
            if start not in g:
                g.add_node(start, lat=seg.start_point.lat, lon=seg.start_point.lon)
            if end not in g:
                g.add_node(end, lat=seg.end_point.lat, lon=seg.end_point.lon)

            # Add edge (allows multiple edges between same nodes)
            g.add_edge(start, end,
                       seg_id=seg.seg_id,
                       seg_name=seg.seg_name,
                       length_mi=seg.length_mi,
                       key=seg.seg_id)

        return g

    def _find_odd_degree_nodes(self, g: nx.MultiGraph) -> list[str]:
        """Find all nodes with odd degree."""
        return [n for n in g.nodes() if g.degree(n) % 2 == 1]

    def _shortest_path_between(self, g: nx.MultiGraph, u: str, v: str) -> tuple[float, list]:
        """Find shortest path between two nodes."""
        try:
            path = nx.shortest_path(g, u, v, weight='length_mi')
            length = nx.shortest_path_length(g, u, v, weight='length_mi')
            return length, path
        except nx.NetworkXNoPath:
            return float('inf'), []

    def _solve_minimum_weight_matching(self, g: nx.MultiGraph, odd_nodes: list[str]) -> list[tuple]:
        """Find minimum weight perfect matching of odd-degree nodes.

        This is the key step in the Chinese Postman Problem.
        We need to pair up odd-degree nodes such that the total distance
        of paths between pairs is minimized.
        """
        if len(odd_nodes) == 0:
            return []

        if len(odd_nodes) == 2:
            return [(odd_nodes[0], odd_nodes[1])]

        # Build complete graph of odd nodes with shortest path distances
        complete = nx.Graph()
        for u in odd_nodes:
            complete.add_node(u)

        for i, u in enumerate(odd_nodes):
            for v in odd_nodes[i + 1:]:
                dist, _ = self._shortest_path_between(g, u, v)
                if dist < float('inf'):
                    complete.add_edge(u, v, weight=dist)

        # Find minimum weight matching
        # networkx has this built-in
        try:
            matching = nx.algorithms.matching.min_weight_matching(complete, weight='weight')
            return list(matching)
        except Exception:
            # Fallback: greedy matching
            matched = set()
            result = []
            for u in odd_nodes:
                if u in matched:
                    continue
                best_v = None
                best_dist = float('inf')
                for v in odd_nodes:
                    if v != u and v not in matched:
                        dist, _ = self._shortest_path_between(g, u, v)
                        if dist < best_dist:
                            best_dist = dist
                            best_v = v
                if best_v:
                    result.append((u, best_v))
                    matched.add(u)
                    matched.add(best_v)
            return result

    def _solve_chinese_postman(self, g: nx.MultiGraph) -> tuple[float, float, list]:
        """Solve Chinese Postman Problem for a connected component.

        Returns: (challenge_miles, garbage_miles, edge_sequence)
        """
        if g.number_of_edges() == 0:
            return 0, 0, []

        # Total challenge miles (each edge once)
        challenge_miles = sum(d['length_mi'] for u, v, d in g.edges(data=True))

        # Find odd-degree nodes
        odd_nodes = self._find_odd_degree_nodes(g)

        # If no odd nodes, we have an Eulerian circuit - perfect efficiency!
        if len(odd_nodes) == 0:
            return challenge_miles, 0, list(nx.eulerian_circuit(g))

        # Find minimum weight matching of odd nodes
        matching = self._solve_minimum_weight_matching(g, odd_nodes)

        # Calculate garbage miles (paths between matched pairs)
        garbage_miles = 0
        augmented = g.copy()

        for u, v in matching:
            dist, path = self._shortest_path_between(g, u, v)
            garbage_miles += dist

            # Add duplicate edges along the path
            for i in range(len(path) - 1):
                # Get existing edge data
                edge_data = g.get_edge_data(path[i], path[i + 1])
                if edge_data:
                    # Use the shortest edge between these nodes
                    min_edge = min(edge_data.values(), key=lambda x: x['length_mi'])
                    augmented.add_edge(path[i], path[i + 1],
                                      seg_id=min_edge['seg_id'],
                                      seg_name=min_edge['seg_name'] + ' (return)',
                                      length_mi=min_edge['length_mi'],
                                      is_duplicate=True)

        # Now we should have an Eulerian circuit
        try:
            circuit = list(nx.eulerian_circuit(augmented))
        except nx.NetworkXError:
            # Graph may be disconnected, get what we can
            circuit = []

        return challenge_miles, garbage_miles, circuit

    def _get_area_name(self, segment_ids: set[int]) -> str:
        """Generate area name from trail names."""
        names = set()
        for seg_id in segment_ids:
            seg = self._segments.get(seg_id)
            if seg:
                parts = seg.seg_name.rsplit(' ', 1)
                base = parts[0] if len(parts) > 1 and parts[1].isdigit() else seg.seg_name
                names.add(base)
        sorted_names = sorted(names)[:3]
        return " / ".join(sorted_names) if sorted_names else "Trail Area"

    def _categorize(self, hours: float) -> HikeCategory:
        if hours <= 2:
            return HikeCategory.SHORT
        elif hours <= 4:
            return HikeCategory.MEDIUM
        else:
            return HikeCategory.LONG

    def _find_best_start_node(self, component: nx.MultiGraph) -> tuple[str, Coordinate]:
        """Find the best starting node (closest to home)."""
        best_node = None
        best_dist = float('inf')

        for node in component.nodes():
            data = component.nodes[node]
            coord = Coordinate(lat=data['lat'], lon=data['lon'])
            dist = haversine_distance(self.home_coord, coord)
            if dist < best_dist:
                best_dist = dist
                best_node = node
                best_coord = coord

        return best_node, best_coord

    def _find_connected_subgraph(self, g: nx.MultiGraph, start_node: str, max_miles: float) -> set:
        """Find a connected subset of edges that fits within max_miles, starting from start_node."""
        selected_edges = set()  # (u, v, key)
        selected_miles = 0
        visited_nodes = {start_node}

        # BFS to find connected edges
        frontier = [start_node]

        while frontier and selected_miles < max_miles:
            node = frontier.pop(0)

            for neighbor in g.neighbors(node):
                for key, data in g.get_edge_data(node, neighbor).items():
                    edge_id = (min(node, neighbor), max(node, neighbor), data['seg_id'])
                    if edge_id not in selected_edges:
                        if selected_miles + data['length_mi'] <= max_miles * 1.2:  # Allow some slack
                            selected_edges.add(edge_id)
                            selected_miles += data['length_mi']
                            if neighbor not in visited_nodes:
                                visited_nodes.add(neighbor)
                                frontier.append(neighbor)

        return selected_edges

    def _split_circuit_into_hikes(
        self,
        component: nx.MultiGraph,
        challenge_miles: float,
        garbage_miles: float
    ) -> list[PlannedHike]:
        """Split a component into day-hike sized chunks using geographic clustering."""
        total_miles = challenge_miles + garbage_miles
        hours = total_miles / self.hiking_pace_mph

        # Collect all segments
        segment_ids = set()
        for u, v, data in component.edges(data=True):
            segment_ids.add(data['seg_id'])

        start_node, start_coord = self._find_best_start_node(component)

        # If small enough, make one hike
        if hours <= self.MAX_HIKE_HOURS:
            segments = []
            seen_segs = set()
            for u, v, data in component.edges(data=True):
                seg_id = data['seg_id']
                is_challenge = seg_id not in seen_segs
                seen_segs.add(seg_id)
                segments.append(HikeSegment(
                    seg_id=seg_id,
                    seg_name=data['seg_name'],
                    length_mi=data['length_mi'],
                    is_challenge=is_challenge
                ))

            return [PlannedHike(
                hike_id=1,
                category=self._categorize(hours),
                area_name=self._get_area_name(segment_ids),
                parking_location=f"Trailhead ({start_coord.lat:.4f}, {start_coord.lon:.4f})",
                parking_lat=start_coord.lat,
                parking_lon=start_coord.lon,
                segments=segments,
                segment_ids=segment_ids,
                challenge_miles=round(challenge_miles, 2),
                garbage_miles=round(garbage_miles, 2),
                estimated_hours=round(hours, 1),
                hiking_pace_mph=self.hiking_pace_mph,
            )]

        # For larger components, use geographic clustering
        # Target ~10 miles challenge per hike (leaves room for garbage)
        target_challenge_mi = min(10, self.MAX_HIKE_HOURS * self.hiking_pace_mph * 0.6)

        hikes = []
        remaining_graph = component.copy()
        seen_globally = set()

        while remaining_graph.number_of_edges() > 0:
            # Find best starting point for this hike (closest to home among remaining)
            best_node = None
            best_dist = float('inf')
            for node in remaining_graph.nodes():
                if remaining_graph.degree(node) > 0:
                    data = remaining_graph.nodes[node]
                    dist = haversine_distance(self.home_coord, Coordinate(lat=data['lat'], lon=data['lon']))
                    if dist < best_dist:
                        best_dist = dist
                        best_node = node

            if not best_node:
                break

            # Collect edges for this hike using BFS from best_node
            hike_edges = []
            hike_challenge_mi = 0
            hike_garbage_mi = 0
            hike_seg_ids = set()
            visited = {best_node}
            frontier = [best_node]

            while frontier and hike_challenge_mi < target_challenge_mi:
                node = frontier.pop(0)

                # Get all edges from this node
                edges_from_node = list(remaining_graph.edges(node, data=True, keys=True))
                # Sort by length (do shorter segments first for better packing)
                edges_from_node.sort(key=lambda e: e[3]['length_mi'])

                for u, v, key, data in edges_from_node:
                    other = v if u == node else u
                    seg_id = data['seg_id']
                    seg_mi = data['length_mi']

                    # Check if adding this would exceed our target too much
                    if hike_challenge_mi + seg_mi > target_challenge_mi * 1.5 and hike_edges:
                        continue

                    is_challenge = seg_id not in seen_globally

                    hike_edges.append((u, v, key, data, is_challenge))
                    if is_challenge:
                        hike_challenge_mi += seg_mi
                        hike_seg_ids.add(seg_id)
                        seen_globally.add(seg_id)
                    else:
                        hike_garbage_mi += seg_mi

                    if other not in visited:
                        visited.add(other)
                        frontier.append(other)

            if not hike_edges:
                break

            # Remove used edges from remaining graph
            for u, v, key, data, _ in hike_edges:
                if remaining_graph.has_edge(u, v, key):
                    remaining_graph.remove_edge(u, v, key)

            # Remove isolated nodes
            remaining_graph.remove_nodes_from(list(nx.isolates(remaining_graph)))

            # Create hike
            segments = [
                HikeSegment(
                    seg_id=data['seg_id'],
                    seg_name=data['seg_name'],
                    length_mi=data['length_mi'],
                    is_challenge=is_chal
                )
                for u, v, key, data, is_chal in hike_edges
            ]

            # Find parking location
            if hike_seg_ids:
                avg_lat = sum(self._segments[s].start_point.lat for s in hike_seg_ids if s in self._segments) / max(1, len(hike_seg_ids))
                avg_lon = sum(self._segments[s].start_point.lon for s in hike_seg_ids if s in self._segments) / max(1, len(hike_seg_ids))
            else:
                data = component.nodes[best_node]
                avg_lat, avg_lon = data['lat'], data['lon']

            total_mi = hike_challenge_mi + hike_garbage_mi
            hike_hours = total_mi / self.hiking_pace_mph

            hikes.append(PlannedHike(
                hike_id=len(hikes) + 1,
                category=self._categorize(hike_hours),
                area_name=self._get_area_name(hike_seg_ids),
                parking_location=f"Trailhead ({avg_lat:.4f}, {avg_lon:.4f})",
                parking_lat=avg_lat,
                parking_lon=avg_lon,
                segments=segments,
                segment_ids=hike_seg_ids.copy(),
                challenge_miles=round(hike_challenge_mi, 2),
                garbage_miles=round(hike_garbage_mi, 2),
                estimated_hours=round(hike_hours, 1),
                hiking_pace_mph=self.hiking_pace_mph,
            ))

        return hikes

    def optimize(self, completed_segments: set[int] | None = None) -> OptimizationResult:
        """Generate optimized hike plan using Chinese Postman approach."""
        completed = completed_segments or set()

        # Build graph excluding completed segments
        g = nx.MultiGraph()
        for seg in self.graph.trail_data.segments:
            if seg.seg_id in completed:
                continue

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

        if g.number_of_edges() == 0:
            return OptimizationResult(hikes=[])

        # Process each connected component separately
        all_hikes = []
        total_challenge = 0
        total_garbage = 0

        for component_nodes in nx.connected_components(g):
            component = g.subgraph(component_nodes).copy()

            challenge_mi, garbage_mi, _ = self._solve_chinese_postman(component)
            total_challenge += challenge_mi
            total_garbage += garbage_mi

            hikes = self._split_circuit_into_hikes(component, challenge_mi, garbage_mi)
            all_hikes.extend(hikes)

        # Sort by distance from home
        def hike_home_dist(h):
            return haversine_distance(self.home_coord, Coordinate(lat=h.parking_lat, lon=h.parking_lon))

        all_hikes.sort(key=hike_home_dist)

        # Renumber
        for i, hike in enumerate(all_hikes):
            hike.hike_id = i + 1

        return OptimizationResult(
            hikes=all_hikes,
            total_challenge_miles=round(total_challenge, 2),
            total_garbage_miles=round(total_garbage, 2),
            total_hikes=len(all_hikes),
            short_hikes=sum(1 for h in all_hikes if h.category == HikeCategory.SHORT),
            medium_hikes=sum(1 for h in all_hikes if h.category == HikeCategory.MEDIUM),
            long_hikes=sum(1 for h in all_hikes if h.category == HikeCategory.LONG),
        )

    def get_hike_for_time(
        self,
        available_hours: float,
        completed_segments: set[int] | None = None
    ) -> PlannedHike | None:
        """Get a single optimized hike for available time budget."""
        result = self.optimize(completed_segments)

        if not result.hikes:
            return None

        # Find best hike that fits in available time
        max_hours = min(available_hours, self.MAX_HIKE_HOURS)

        fitting_hikes = [h for h in result.hikes if h.estimated_hours <= max_hours]
        if fitting_hikes:
            # Return closest one
            return fitting_hikes[0]

        # No hikes fit - return smallest one
        return min(result.hikes, key=lambda h: h.estimated_hours)


def create_optimizer_v4(
    trail_graph: TrailGraph,
    home_lat: float,
    home_lon: float,
    hiking_pace_mph: float = 2.5
) -> ChinesePostmanOptimizer:
    """Create a v4 Chinese Postman optimizer."""
    return ChinesePostmanOptimizer(trail_graph, home_lat, home_lon, hiking_pace_mph)
