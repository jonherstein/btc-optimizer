"""Route optimization v3 - Graph-based optimization with OR-Tools.

Uses actual shortest paths through the trail network and OR-Tools CVRP solver
to minimize total garbage miles and driving distance.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from typing import Optional
import heapq

import networkx as nx
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from backend.models.trail import Coordinate, TrailSegment
from backend.services.graph_builder import TrailGraph, haversine_distance


class HikeCategory(str, Enum):
    SHORT = "short"      # 1-2 hours
    MEDIUM = "medium"    # 2-4 hours
    LONG = "long"        # 4-6 hours


@dataclass
class RouteSegment:
    """A segment in a planned hiking route."""
    seg_id: int
    seg_name: str
    length_mi: float
    is_challenge: bool  # True = counts toward BTC challenge
    from_node: str
    to_node: str


@dataclass
class PlannedHike:
    """A complete hike with actual route through trail network."""
    hike_id: int
    category: HikeCategory
    area_name: str

    # Car logistics
    parking_location: str
    parking_lat: float
    parking_lon: float
    parking_node: str

    # Actual route (ordered list of segments to hike)
    route: list[RouteSegment] = field(default_factory=list)
    challenge_segment_ids: set[int] = field(default_factory=set)

    # Miles breakdown (calculated from route)
    challenge_miles: float = 0.0
    garbage_miles: float = 0.0

    # Time
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

    # For compatibility with v2 interface
    @property
    def segments(self) -> list:
        """Convert route to HikeSegment-like objects for API compatibility."""
        from backend.services.route_optimizer_v2 import HikeSegment
        result = []
        for rs in self.route:
            result.append(HikeSegment(
                seg_id=rs.seg_id,
                seg_name=rs.seg_name,
                length_mi=rs.length_mi,
                is_challenge=rs.is_challenge
            ))
        return result

    @property
    def segment_ids(self) -> set[int]:
        return self.challenge_segment_ids


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


class GraphOptimizer:
    """Optimizes hiking routes using graph algorithms and OR-Tools."""

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

        # Build undirected walking graph (for pathfinding)
        self._walking_graph = self._build_walking_graph()

        # Precompute segment info
        self._segments = {s.seg_id: s for s in self.graph.trail_data.segments}
        self._segment_nodes = {}  # seg_id -> (start_node, end_node)
        self._node_to_segments = defaultdict(set)  # node_id -> set of seg_ids

        for seg in self.graph.trail_data.segments:
            start = self._find_node_for_coord(seg.start_point)
            end = self._find_node_for_coord(seg.end_point)
            self._segment_nodes[seg.seg_id] = (start, end)
            self._node_to_segments[start].add(seg.seg_id)
            self._node_to_segments[end].add(seg.seg_id)

        # Cache shortest paths
        self._path_cache = {}

    def _build_walking_graph(self) -> nx.Graph:
        """Build undirected graph for walking distances."""
        g = nx.Graph()
        for node_id in self.graph.nodes:
            g.add_node(node_id)

        # Add edges from the trail graph
        for u, v, data in self.graph.graph.edges(data=True):
            length_mi = data.get("length_mi", 0)
            seg_id = data.get("seg_id")
            if g.has_edge(u, v):
                # Keep shortest if multiple edges
                if g[u][v]["weight"] > length_mi:
                    g[u][v]["weight"] = length_mi
                    g[u][v]["seg_id"] = seg_id
            else:
                g.add_edge(u, v, weight=length_mi, seg_id=seg_id)

        return g

    def _find_node_for_coord(self, coord: Coordinate) -> str:
        """Find the graph node closest to a coordinate."""
        min_dist = float("inf")
        best_node = None
        for node_id, node_info in self.graph.nodes.items():
            dist = haversine_distance(coord, node_info.coord)
            if dist < min_dist:
                min_dist = dist
                best_node = node_id
        return best_node

    def _shortest_path_distance(self, from_node: str, to_node: str) -> tuple[float, list[str]]:
        """Get shortest path distance and path between two nodes."""
        cache_key = (from_node, to_node)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        try:
            path = nx.shortest_path(self._walking_graph, from_node, to_node, weight="weight")
            dist = nx.shortest_path_length(self._walking_graph, from_node, to_node, weight="weight")
            result = (dist, path)
        except nx.NetworkXNoPath:
            result = (float("inf"), [])

        self._path_cache[cache_key] = result
        return result

    def _drive_distance_miles(self, coord: Coordinate) -> float:
        """Estimate drive distance from home to a point (as the crow flies * 1.4)."""
        direct_m = haversine_distance(self.home_coord, coord)
        return (direct_m / self.METERS_PER_MILE) * 1.4  # Road factor

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

    def _find_best_parking(self, segment_ids: set[int]) -> tuple[str, Coordinate]:
        """Find the best parking node for a set of segments.

        Prefers:
        1. Trailheads (nodes with low degree)
        2. Nodes connected to many of the target segments
        3. Nodes closer to home
        """
        # Collect all nodes touching these segments
        candidate_nodes = set()
        for seg_id in segment_ids:
            if seg_id in self._segment_nodes:
                start, end = self._segment_nodes[seg_id]
                candidate_nodes.add(start)
                candidate_nodes.add(end)

        if not candidate_nodes:
            # Fallback to nearest trailhead
            trailheads = self.graph.get_trailheads()
            if trailheads:
                best = min(trailheads, key=lambda n: haversine_distance(
                    self.home_coord, self.graph.nodes[n].coord
                ))
                return best, self.graph.nodes[best].coord

        # Score each candidate
        best_node = None
        best_score = float("inf")

        for node in candidate_nodes:
            node_info = self.graph.nodes[node]

            # Prefer trailheads
            trailhead_bonus = 0 if node_info.is_trailhead else 2

            # Distance from home (in miles)
            dist_mi = haversine_distance(self.home_coord, node_info.coord) / self.METERS_PER_MILE

            # Connectivity to target segments
            connected = len(self._node_to_segments[node] & segment_ids)
            connectivity_bonus = -connected * 0.5  # Negative = better

            score = dist_mi + trailhead_bonus + connectivity_bonus
            if score < best_score:
                best_score = score
                best_node = node

        return best_node, self.graph.nodes[best_node].coord

    def _compute_route_for_segments(
        self,
        segment_ids: set[int],
        parking_node: str
    ) -> tuple[list[RouteSegment], float, float]:
        """Compute actual hiking route through segments, starting and ending at parking.

        Uses a greedy nearest-neighbor approach with actual path distances.

        Returns: (route, challenge_miles, garbage_miles)
        """
        if not segment_ids:
            return [], 0.0, 0.0

        route = []
        challenge_miles = 0.0
        garbage_miles = 0.0
        visited_challenge = set()

        current_node = parking_node

        # Greedy: always go to nearest unvisited challenge segment
        remaining = set(segment_ids)

        while remaining:
            best_seg = None
            best_dist = float("inf")
            best_entry_node = None
            best_exit_node = None

            for seg_id in remaining:
                seg = self._segments[seg_id]
                start_node, end_node = self._segment_nodes[seg_id]

                # Try entering from start
                dist_to_start, _ = self._shortest_path_distance(current_node, start_node)
                if dist_to_start < best_dist:
                    best_dist = dist_to_start
                    best_seg = seg_id
                    best_entry_node = start_node
                    best_exit_node = end_node

                # Try entering from end
                dist_to_end, _ = self._shortest_path_distance(current_node, end_node)
                if dist_to_end < best_dist:
                    best_dist = dist_to_end
                    best_seg = seg_id
                    best_entry_node = end_node
                    best_exit_node = start_node

            if best_seg is None or best_dist == float("inf"):
                break  # Can't reach remaining segments

            seg = self._segments[best_seg]

            # Add connector path if needed (garbage miles)
            if current_node != best_entry_node:
                connector_dist, connector_path = self._shortest_path_distance(
                    current_node, best_entry_node
                )
                if connector_dist > 0 and connector_path:
                    # Add connector segments
                    for i in range(len(connector_path) - 1):
                        u, v = connector_path[i], connector_path[i + 1]
                        edge_data = self._walking_graph.get_edge_data(u, v)
                        if edge_data:
                            edge_seg_id = edge_data.get("seg_id")
                            edge_len = edge_data.get("weight", 0)
                            edge_seg = self._segments.get(edge_seg_id)
                            edge_name = edge_seg.seg_name if edge_seg else "Connector"

                            # Check if this is a challenge segment we haven't done
                            is_challenge = (
                                edge_seg_id in segment_ids and
                                edge_seg_id not in visited_challenge
                            )

                            route.append(RouteSegment(
                                seg_id=edge_seg_id or 0,
                                seg_name=edge_name,
                                length_mi=round(edge_len, 2),
                                is_challenge=is_challenge,
                                from_node=u,
                                to_node=v
                            ))

                            if is_challenge:
                                challenge_miles += edge_len
                                visited_challenge.add(edge_seg_id)
                                remaining.discard(edge_seg_id)
                            else:
                                garbage_miles += edge_len

            # Add the target challenge segment
            if best_seg not in visited_challenge:
                route.append(RouteSegment(
                    seg_id=best_seg,
                    seg_name=seg.seg_name,
                    length_mi=round(seg.length_mi, 2),
                    is_challenge=True,
                    from_node=best_entry_node,
                    to_node=best_exit_node
                ))
                challenge_miles += seg.length_mi
                visited_challenge.add(best_seg)

            remaining.discard(best_seg)
            current_node = best_exit_node

        # Return to parking
        if current_node != parking_node:
            return_dist, return_path = self._shortest_path_distance(current_node, parking_node)
            if return_dist > 0 and return_path:
                for i in range(len(return_path) - 1):
                    u, v = return_path[i], return_path[i + 1]
                    edge_data = self._walking_graph.get_edge_data(u, v)
                    if edge_data:
                        edge_seg_id = edge_data.get("seg_id")
                        edge_len = edge_data.get("weight", 0)
                        edge_seg = self._segments.get(edge_seg_id)
                        edge_name = edge_seg.seg_name if edge_seg else "Return"

                        is_challenge = (
                            edge_seg_id in segment_ids and
                            edge_seg_id not in visited_challenge
                        )

                        route.append(RouteSegment(
                            seg_id=edge_seg_id or 0,
                            seg_name=edge_name,
                            length_mi=round(edge_len, 2),
                            is_challenge=is_challenge,
                            from_node=u,
                            to_node=v
                        ))

                        if is_challenge:
                            challenge_miles += edge_len
                            visited_challenge.add(edge_seg_id)
                        else:
                            garbage_miles += edge_len

        return route, round(challenge_miles, 2), round(garbage_miles, 2)

    def _cluster_segments_by_connectivity(self, segment_ids: set[int]) -> list[set[int]]:
        """Cluster segments based on actual trail connectivity.

        Segments that can be hiked together without excessive travel are grouped.
        """
        if not segment_ids:
            return []

        # Build connectivity graph between segments
        seg_graph = nx.Graph()
        for seg_id in segment_ids:
            seg_graph.add_node(seg_id)

        # Add edges between segments that share a node or are close
        seg_list = list(segment_ids)
        for i, seg_a in enumerate(seg_list):
            nodes_a = set(self._segment_nodes.get(seg_a, ()))
            for seg_b in seg_list[i + 1:]:
                nodes_b = set(self._segment_nodes.get(seg_b, ()))

                # Shared node = directly connected
                if nodes_a & nodes_b:
                    seg_graph.add_edge(seg_a, seg_b, weight=0)
                else:
                    # Check shortest path between closest nodes
                    min_dist = float("inf")
                    for na in nodes_a:
                        for nb in nodes_b:
                            dist, _ = self._shortest_path_distance(na, nb)
                            min_dist = min(min_dist, dist)

                    # Connect if reasonably close (< 2 miles connector)
                    if min_dist < 2.0:
                        seg_graph.add_edge(seg_a, seg_b, weight=min_dist)

        # Find connected components
        return [set(c) for c in nx.connected_components(seg_graph)]

    def _split_cluster_by_time(self, segment_ids: set[int]) -> list[set[int]]:
        """Split a cluster into hikes that fit within time budget."""
        if not segment_ids:
            return []

        total_challenge = sum(self._segments[s].length_mi for s in segment_ids)

        # Conservative estimate: assume 2x challenge miles for total (100% garbage overhead)
        # This is safer for areas with poor connectivity
        estimated_total = total_challenge * 2.0
        estimated_hours = estimated_total / self.hiking_pace_mph

        if estimated_hours <= self.MAX_HIKE_HOURS:
            return [segment_ids]

        # Need to split - use a bin-packing approach
        # Sort segments by geographic proximity to create logical groups
        seg_list = list(segment_ids)

        # Find centroid of all segments
        all_coords = []
        for seg_id in seg_list:
            seg = self._segments[seg_id]
            all_coords.append(seg.start_point)
            all_coords.append(seg.end_point)

        centroid_lat = sum(c.lat for c in all_coords) / len(all_coords)
        centroid_lon = sum(c.lon for c in all_coords) / len(all_coords)
        centroid = Coordinate(lat=centroid_lat, lon=centroid_lon)

        # Sort by angle from centroid (creates a sweep pattern)
        def angle_from_centroid(seg_id):
            seg = self._segments[seg_id]
            mid_lat = (seg.start_point.lat + seg.end_point.lat) / 2
            mid_lon = (seg.start_point.lon + seg.end_point.lon) / 2
            return math.atan2(mid_lat - centroid_lat, mid_lon - centroid_lon)

        seg_list.sort(key=angle_from_centroid)

        # Pack into bins - target 5 challenge miles per hike (very conservative)
        result = []
        current = set()
        current_mi = 0.0

        # Target ~5 challenge miles per hike to stay well under 6 hours
        max_challenge_mi = 5.0

        for seg_id in seg_list:
            seg_mi = self._segments[seg_id].length_mi
            if current_mi + seg_mi > max_challenge_mi and current:
                result.append(current)
                current = set()
                current_mi = 0.0

            current.add(seg_id)
            current_mi += seg_mi

        if current:
            result.append(current)

        return result

    def _create_hike(self, hike_id: int, segment_ids: set[int]) -> PlannedHike:
        """Create a fully-planned hike with actual route."""
        parking_node, parking_coord = self._find_best_parking(segment_ids)
        route, challenge_mi, garbage_mi = self._compute_route_for_segments(
            segment_ids, parking_node
        )

        total_mi = challenge_mi + garbage_mi
        hours = total_mi / self.hiking_pace_mph

        return PlannedHike(
            hike_id=hike_id,
            category=self._categorize(hours),
            area_name=self._get_area_name(segment_ids),
            parking_location=f"Trailhead ({parking_coord.lat:.4f}, {parking_coord.lon:.4f})",
            parking_lat=parking_coord.lat,
            parking_lon=parking_coord.lon,
            parking_node=parking_node,
            route=route,
            challenge_segment_ids=segment_ids,
            challenge_miles=challenge_mi,
            garbage_miles=garbage_mi,
            estimated_hours=round(hours, 1),
            hiking_pace_mph=self.hiking_pace_mph,
        )

    def _split_oversized_hike(self, segment_ids: set[int], max_iterations: int = 5) -> list[set[int]]:
        """Recursively split a segment group until all parts fit within time limit."""
        if len(segment_ids) <= 1:
            return [segment_ids]

        # Create hike and check time
        hike = self._create_hike(0, segment_ids)
        if hike.estimated_hours <= self.MAX_HIKE_HOURS:
            return [segment_ids]

        # Need to split - divide in half geographically
        seg_list = list(segment_ids)
        seg_list.sort(key=lambda s: (
            self._segments[s].start_point.lat + self._segments[s].end_point.lat
        ) / 2)

        mid = len(seg_list) // 2
        first_half = set(seg_list[:mid])
        second_half = set(seg_list[mid:])

        result = []

        # Recursively check each half
        if max_iterations > 0:
            for half in [first_half, second_half]:
                if half:
                    result.extend(self._split_oversized_hike(half, max_iterations - 1))
        else:
            # Hit iteration limit - just return as-is
            if first_half:
                result.append(first_half)
            if second_half:
                result.append(second_half)

        return result

    def optimize(self, completed_segments: set[int] | None = None) -> OptimizationResult:
        """Generate optimized hike plan using graph-based approach."""
        completed = completed_segments or set()
        remaining = set(self._segments.keys()) - completed

        if not remaining:
            return OptimizationResult(hikes=[])

        # Step 1: Cluster by connectivity
        clusters = self._cluster_segments_by_connectivity(remaining)

        # Step 2: Split large clusters by time (initial estimate)
        hike_groups = []
        for cluster in clusters:
            splits = self._split_cluster_by_time(cluster)
            hike_groups.extend(splits)

        # Step 3: Create hikes and compute actual routes
        hikes = []
        for i, group in enumerate(hike_groups):
            hike = self._create_hike(i + 1, group)

            # Step 3b: If hike exceeds limit, split it further
            if hike.estimated_hours > self.MAX_HIKE_HOURS and len(group) > 1:
                sub_groups = self._split_oversized_hike(group)
                for sg in sub_groups:
                    if sg:
                        hikes.append(self._create_hike(len(hikes) + 1, sg))
            else:
                hikes.append(hike)

        # Step 4: Sort by drive distance from home + hiking efficiency
        def hike_score(h: PlannedHike) -> float:
            drive_dist = self._drive_distance_miles(
                Coordinate(lat=h.parking_lat, lon=h.parking_lon)
            )
            # Prioritize: close to home + high efficiency
            return drive_dist - (h.efficiency / 50)

        hikes.sort(key=hike_score)

        # Renumber
        for i, hike in enumerate(hikes):
            hike.hike_id = i + 1

        # Calculate totals
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

    def optimize_with_ortools(self, completed_segments: set[int] | None = None) -> OptimizationResult:
        """Optimize using OR-Tools CVRP solver for global minimum garbage miles.

        This treats the problem as a Capacitated Vehicle Routing Problem where:
        - Depot = home
        - Customers = segment clusters (groups of nearby segments)
        - Vehicle capacity = max hike hours
        - Distance = garbage miles between clusters
        """
        completed = completed_segments or set()
        remaining = set(self._segments.keys()) - completed

        if not remaining:
            return OptimizationResult(hikes=[])

        # First cluster into small groups that can be hiked together
        micro_clusters = self._cluster_segments_by_connectivity(remaining)

        # Further split any large micro-clusters
        cluster_list = []
        for mc in micro_clusters:
            splits = self._split_cluster_by_time(mc)
            cluster_list.extend(splits)

        if len(cluster_list) < 2:
            # Too few clusters for VRP, use simple approach
            return self.optimize(completed_segments)

        n = len(cluster_list)

        # Compute distance matrix between cluster centroids
        cluster_coords = []
        for cluster in cluster_list:
            coords = []
            for seg_id in cluster:
                seg = self._segments[seg_id]
                coords.append(seg.start_point)
                coords.append(seg.end_point)
            avg_lat = sum(c.lat for c in coords) / len(coords)
            avg_lon = sum(c.lon for c in coords) / len(coords)
            cluster_coords.append(Coordinate(lat=avg_lat, lon=avg_lon))

        # Add home as depot (index 0)
        all_coords = [self.home_coord] + cluster_coords

        # Build distance matrix (in units of 100m for integer math)
        def dist_units(c1: Coordinate, c2: Coordinate) -> int:
            return int(haversine_distance(c1, c2) / 100)

        distance_matrix = []
        for i, c1 in enumerate(all_coords):
            row = []
            for j, c2 in enumerate(all_coords):
                if i == j:
                    row.append(0)
                else:
                    row.append(dist_units(c1, c2))
            distance_matrix.append(row)

        # Estimate time for each cluster (in minutes)
        cluster_times = [0]  # Depot has 0 time
        for cluster in cluster_list:
            challenge_mi = sum(self._segments[s].length_mi for s in cluster)
            estimated_mi = challenge_mi * 1.5  # Add garbage estimate
            time_min = int((estimated_mi / self.hiking_pace_mph) * 60)
            cluster_times.append(time_min)

        # Create routing model
        manager = pywrapcp.RoutingIndexManager(
            len(distance_matrix),
            10,  # Up to 10 vehicles (hikes)
            0    # Depot at index 0
        )
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add time capacity constraint
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            # Time = cluster hiking time + travel time between clusters
            travel_min = int((distance_matrix[from_node][to_node] * 100 / self.METERS_PER_MILE) / self.hiking_pace_mph * 60)
            return cluster_times[to_node] + travel_min

        time_callback_index = routing.RegisterTransitCallback(time_callback)

        max_time_min = int(self.MAX_HIKE_HOURS * 60)
        routing.AddDimensionWithVehicleCapacity(
            time_callback_index,
            0,  # No slack
            [max_time_min] * 10,  # Vehicle max capacities
            True,  # Start cumul to zero
            "Time"
        )

        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 30  # Allow up to 30s for optimization

        # Solve
        solution = routing.SolveWithParameters(search_parameters)

        if not solution:
            # Fallback to simple approach
            return self.optimize(completed_segments)

        # Extract routes from solution
        hikes = []
        hike_id = 1

        for vehicle_id in range(10):
            index = routing.Start(vehicle_id)
            route_clusters = []

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node > 0:  # Skip depot
                    route_clusters.append(cluster_list[node - 1])
                index = solution.Value(routing.NextVar(index))

            if route_clusters:
                # Combine clusters for this vehicle into one hike
                combined = set()
                for rc in route_clusters:
                    combined.update(rc)

                hike = self._create_hike(hike_id, combined)
                hikes.append(hike)
                hike_id += 1

        # Sort by proximity to home
        hikes.sort(key=lambda h: haversine_distance(
            self.home_coord, Coordinate(lat=h.parking_lat, lon=h.parking_lon)
        ))

        # Renumber
        for i, hike in enumerate(hikes):
            hike.hike_id = i + 1

        # Calculate totals
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
        """Get a single optimized hike for available time budget."""
        completed = completed_segments or set()
        remaining = set(self._segments.keys()) - completed

        if not remaining:
            return None

        max_hours = min(available_hours, self.MAX_HIKE_HOURS)

        # Cluster remaining segments
        clusters = self._cluster_segments_by_connectivity(remaining)

        if not clusters:
            return None

        # Sort clusters by proximity to home
        def cluster_home_dist(cluster: set[int]) -> float:
            coords = []
            for seg_id in cluster:
                seg = self._segments[seg_id]
                coords.append(seg.start_point)
                coords.append(seg.end_point)
            avg_lat = sum(c.lat for c in coords) / len(coords)
            avg_lon = sum(c.lon for c in coords) / len(coords)
            return haversine_distance(self.home_coord, Coordinate(lat=avg_lat, lon=avg_lon))

        clusters.sort(key=cluster_home_dist)

        # Find best fit from closest cluster
        best_cluster = clusters[0]
        max_mi = (max_hours * self.hiking_pace_mph) / 1.5  # Account for garbage

        # Select segments that fit
        sorted_segs = sorted(
            best_cluster,
            key=lambda s: self._segments[s].length_mi
        )

        selected = set()
        selected_mi = 0.0

        for seg_id in sorted_segs:
            seg_mi = self._segments[seg_id].length_mi
            if selected_mi + seg_mi <= max_mi:
                selected.add(seg_id)
                selected_mi += seg_mi

        if not selected and sorted_segs:
            selected.add(sorted_segs[0])

        hike = self._create_hike(1, selected)

        # Verify it fits
        while hike.estimated_hours > max_hours and len(selected) > 1:
            largest = max(selected, key=lambda s: self._segments[s].length_mi)
            selected.remove(largest)
            hike = self._create_hike(1, selected)

        return hike


def create_optimizer_v3(
    trail_graph: TrailGraph,
    home_lat: float,
    home_lon: float,
    hiking_pace_mph: float = 2.5,
    use_ortools: bool = True
) -> GraphOptimizer:
    """Create a v3 graph-based optimizer."""
    return GraphOptimizer(trail_graph, home_lat, home_lon, hiking_pace_mph)
