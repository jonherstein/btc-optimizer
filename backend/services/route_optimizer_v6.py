"""Route optimization v6 - Chinese Postman + Shuttle-aware optimization.

Combines the optimal Chinese Postman solution with shuttle recommendations:
1. Uses Chinese Postman to find minimum garbage miles needed
2. Splits into practical day-hike sized chunks (8-12 challenge miles each)
3. Analyzes each hike to determine if shuttle would save significant miles
4. Recommends shuttle when it saves > 1 mile

Based on research from BTC finishers:
- Even efficient finishers do 200-230 miles for 177-mile challenge (13-30% overhead)
- "Getting rides to trailheads was the most important tool" - 2021 finisher
- "A big part of this challenge is maximizing your routes" - participant advice
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

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
    is_challenge: bool  # True = counts toward challenge


@dataclass
class PlannedHike:
    """A complete hike plan with optional shuttle support."""
    hike_id: int
    category: HikeCategory
    area_name: str

    # Start location (where to park / get dropped off)
    start_location: str
    start_lat: float
    start_lon: float

    # End location (same as start for loops, different for shuttles)
    end_location: str
    end_lat: float
    end_lon: float

    # Shuttle info
    needs_shuttle: bool = False
    shuttle_savings_miles: float = 0.0
    shuttle_note: str = ""

    # For backwards compatibility
    @property
    def parking_location(self) -> str:
        return self.start_location

    @property
    def parking_lat(self) -> float:
        return self.start_lat

    @property
    def parking_lon(self) -> float:
        return self.start_lon

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
    shuttle_hikes: int = 0
    total_shuttle_savings: float = 0.0

    @property
    def total_miles(self) -> float:
        return self.total_challenge_miles + self.total_garbage_miles

    @property
    def overall_efficiency(self) -> float:
        if self.total_miles == 0:
            return 0
        return (self.total_challenge_miles / self.total_miles) * 100


class ShuttleAwareOptimizer:
    """Optimizes hiking routes using Chinese Postman + shuttle recommendations."""

    MAX_HIKE_HOURS = 6.0
    TARGET_CHALLENGE_MILES = 10.0  # Ideal challenge miles per hike
    MAX_CHALLENGE_MILES = 12.0     # Maximum before splitting
    SHUTTLE_THRESHOLD_MILES = 1.5  # Recommend shuttle if saves > this

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
        self._trail_graph = self._build_trail_graph()
        self._dead_ends = self._find_dead_ends()

    def _build_trail_graph(self) -> nx.MultiGraph:
        """Build undirected multigraph from trail segments."""
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

    def _find_dead_ends(self) -> set[str]:
        """Find all dead-end nodes (degree 1)."""
        return {n for n in self._trail_graph.nodes() if self._trail_graph.degree(n) == 1}

    def _get_node_coord(self, node: str) -> Coordinate:
        """Get coordinate for a node."""
        data = self._trail_graph.nodes[node]
        return Coordinate(lat=data['lat'], lon=data['lon'])

    def _find_odd_degree_nodes(self, g: nx.MultiGraph) -> list[str]:
        """Find all nodes with odd degree."""
        return [n for n in g.nodes() if g.degree(n) % 2 == 1]

    def _shortest_path_length(self, g: nx.MultiGraph, u: str, v: str) -> float:
        """Get shortest path length between two nodes."""
        try:
            return nx.shortest_path_length(g, u, v, weight='length_mi')
        except nx.NetworkXNoPath:
            return float('inf')

    def _solve_minimum_weight_matching(self, g: nx.MultiGraph, odd_nodes: list[str]) -> list[tuple]:
        """Find minimum weight perfect matching of odd-degree nodes."""
        if len(odd_nodes) == 0:
            return []
        if len(odd_nodes) == 2:
            return [(odd_nodes[0], odd_nodes[1])]

        # Build complete graph of odd nodes
        complete = nx.Graph()
        for u in odd_nodes:
            complete.add_node(u)

        for i, u in enumerate(odd_nodes):
            for v in odd_nodes[i + 1:]:
                dist = self._shortest_path_length(g, u, v)
                if dist < float('inf'):
                    complete.add_edge(u, v, weight=dist)

        try:
            matching = nx.algorithms.matching.min_weight_matching(complete, weight='weight')
            return list(matching)
        except Exception:
            # Greedy fallback
            matched = set()
            result = []
            for u in odd_nodes:
                if u in matched:
                    continue
                best_v = None
                best_dist = float('inf')
                for v in odd_nodes:
                    if v != u and v not in matched:
                        dist = self._shortest_path_length(g, u, v)
                        if dist < best_dist:
                            best_dist = dist
                            best_v = v
                if best_v:
                    result.append((u, best_v))
                    matched.add(u)
                    matched.add(best_v)
            return result

    def _solve_chinese_postman(self, g: nx.MultiGraph) -> tuple[float, float]:
        """Solve Chinese Postman Problem for a graph.

        Returns: (challenge_miles, garbage_miles)
        """
        if g.number_of_edges() == 0:
            return 0, 0

        challenge_miles = sum(d['length_mi'] for u, v, d in g.edges(data=True))
        odd_nodes = self._find_odd_degree_nodes(g)

        if len(odd_nodes) == 0:
            return challenge_miles, 0

        matching = self._solve_minimum_weight_matching(g, odd_nodes)

        garbage_miles = 0
        for u, v in matching:
            dist = self._shortest_path_length(g, u, v)
            if dist < float('inf'):
                garbage_miles += dist

        return challenge_miles, garbage_miles

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

    def _find_best_trailhead(self, g: nx.MultiGraph, segment_ids: set[int]) -> tuple[str, Coordinate]:
        """Find the best trailhead for a set of segments."""
        # Collect nodes from these segments
        nodes = set()
        for u, v, data in g.edges(data=True):
            if data['seg_id'] in segment_ids:
                nodes.add(u)
                nodes.add(v)

        if not nodes:
            return "Home", self.home_coord

        # Prefer dead-ends (often parking areas) or high-degree junctions
        best_node = None
        best_score = float('inf')

        for node in nodes:
            dist = haversine_distance(self.home_coord, self._get_node_coord(node))
            degree = g.degree(node)

            # Score: closer to home is better, dead-ends and junctions are preferred
            if degree == 1 or degree >= 3:
                score = dist * 0.8  # 20% bonus for good trailheads
            else:
                score = dist

            if score < best_score:
                best_score = score
                best_node = node

        if best_node:
            coord = self._get_node_coord(best_node)
            return f"Trailhead ({coord.lat:.4f}, {coord.lon:.4f})", coord

        return "Home", self.home_coord

    def _compute_walking_order(
        self,
        subgraph: nx.MultiGraph,
        segment_ids: set[int],
        start_node: str,
        end_node: str,
        seen_globally: set[int]
    ) -> list[HikeSegment]:
        """Compute actual walking order through segments using DFS traversal.

        Uses a modified DFS that visits all edges, backtracking as needed.
        Returns ordered list of HikeSegment with proper is_challenge flags.
        """
        if not segment_ids or len(subgraph.edges()) == 0:
            return []

        # Build edge tracking - need to visit each edge at least once
        edges_to_visit = set()
        edge_lookup = {}  # (u, v, key) -> edge_data
        for u, v, key, data in subgraph.edges(keys=True, data=True):
            edges_to_visit.add((u, v, key))
            edge_lookup[(u, v, key)] = data
            edge_lookup[(v, u, key)] = data  # Can traverse either direction

        # Find best starting node (prefer start_node if it exists)
        if start_node in subgraph.nodes():
            current = start_node
        else:
            current = next(iter(subgraph.nodes()))

        # DFS to visit all edges
        segments_list = []
        seen_in_hike = set()
        visited_edges = set()

        def get_unvisited_edges_from(node):
            """Get edges from this node that haven't been visited."""
            result = []
            for u, v, key, data in subgraph.edges(node, keys=True, data=True):
                edge_key = tuple(sorted([u, v])) + (key,)
                if edge_key not in visited_edges:
                    result.append((u, v, key, data))
            return result

        def mark_visited(u, v, key):
            edge_key = tuple(sorted([u, v])) + (key,)
            visited_edges.add(edge_key)

        def is_visited(u, v, key):
            edge_key = tuple(sorted([u, v])) + (key,)
            return edge_key in visited_edges

        # Main traversal loop
        path_stack = [current]
        total_edges = len(subgraph.edges())

        while len(visited_edges) < total_edges:
            # Try to find an unvisited edge from current node
            unvisited = get_unvisited_edges_from(current)

            if unvisited:
                # Take an unvisited edge (prefer challenge segments first)
                unvisited.sort(key=lambda e: (e[3].get('seg_id', 0) not in segment_ids, e[3].get('length_mi', 0)))
                u, v, key, data = unvisited[0]
                next_node = v if u == current else u
                mark_visited(u, v, key)

                # Add segment to list
                seg_id = data.get('seg_id', -1)
                seg_name = data.get('seg_name', '')
                length_mi = data.get('length_mi', 0)

                is_challenge = (
                    seg_id in segment_ids and
                    seg_id not in seen_globally and
                    seg_id not in seen_in_hike
                )
                if is_challenge:
                    seen_in_hike.add(seg_id)

                segments_list.append(HikeSegment(
                    seg_id=seg_id,
                    seg_name=seg_name,
                    length_mi=length_mi,
                    is_challenge=is_challenge
                ))

                path_stack.append(next_node)
                current = next_node

            elif len(path_stack) > 1:
                # Backtrack - need to walk back along a previously visited edge
                path_stack.pop()
                prev_node = path_stack[-1]

                # Find the edge back (already visited, so this is "garbage" traversal)
                for u, v, key, data in subgraph.edges(current, keys=True, data=True):
                    next_node = v if u == current else u
                    if next_node == prev_node:
                        seg_id = data.get('seg_id', -1)
                        seg_name = data.get('seg_name', '')
                        length_mi = data.get('length_mi', 0)

                        # This is a backtrack, so it's garbage miles (not challenge)
                        segments_list.append(HikeSegment(
                            seg_id=seg_id,
                            seg_name=seg_name + " (return)",
                            length_mi=length_mi,
                            is_challenge=False
                        ))
                        break

                current = prev_node

            else:
                # No more edges reachable - find any node with unvisited edges
                found = False
                for node in subgraph.nodes():
                    if get_unvisited_edges_from(node):
                        # Need to pathfind to this node
                        try:
                            path = nx.shortest_path(subgraph, current, node)
                            # Walk the path (all edges already visited, so garbage)
                            for i in range(len(path) - 1):
                                a, b = path[i], path[i + 1]
                                edge_data_dict = subgraph.get_edge_data(a, b)
                                if edge_data_dict:
                                    ekey, edata = next(iter(edge_data_dict.items()))
                                    segments_list.append(HikeSegment(
                                        seg_id=edata.get('seg_id', -1),
                                        seg_name=edata.get('seg_name', '') + " (connector)",
                                        length_mi=edata.get('length_mi', 0),
                                        is_challenge=False
                                    ))
                            current = node
                            path_stack = [node]
                            found = True
                            break
                        except nx.NetworkXNoPath:
                            continue
                if not found:
                    break  # Graph is disconnected, give up

        return segments_list

    def _analyze_shuttle_benefit(
        self,
        g: nx.MultiGraph,
        segment_ids: set[int],
        start_node: str
    ) -> tuple[bool, float, str, Coordinate]:
        """Analyze if using a shuttle would save significant miles.

        Returns: (needs_shuttle, savings, end_node, end_coord)
        """
        # Build subgraph
        nodes_in_hike = set()
        for u, v, data in g.edges(data=True):
            if data['seg_id'] in segment_ids:
                nodes_in_hike.add(u)
                nodes_in_hike.add(v)

        if len(nodes_in_hike) < 2:
            start_coord = self._get_node_coord(start_node)
            return False, 0, start_node, start_coord

        # Find the node furthest from start that's a good pickup point
        best_end = start_node
        max_backtrack_saved = 0

        for node in nodes_in_hike:
            if node == start_node:
                continue

            # Distance you'd have to backtrack in a loop
            backtrack = self._shortest_path_length(g, node, start_node)
            if backtrack == float('inf'):
                continue

            # Only consider if it's a reasonable pickup point
            degree = g.degree(node)
            if degree == 1 or degree >= 3:  # Dead-end or junction
                if backtrack > max_backtrack_saved:
                    max_backtrack_saved = backtrack
                    best_end = node

        if max_backtrack_saved > self.SHUTTLE_THRESHOLD_MILES:
            end_coord = self._get_node_coord(best_end)
            return True, max_backtrack_saved, best_end, end_coord

        start_coord = self._get_node_coord(start_node)
        return False, 0, start_node, start_coord

    def _create_hike(
        self,
        hike_id: int,
        g: nx.MultiGraph,
        segment_ids: set[int],
        seen_globally: set[int]
    ) -> PlannedHike:
        """Create a hike plan for a set of segments."""
        if not segment_ids:
            return None

        # Find start trailhead
        start_name, start_coord = self._find_best_trailhead(g, segment_ids)
        start_node = f"{start_coord.lat:.5f},{start_coord.lon:.5f}"

        # Calculate challenge and garbage miles
        challenge_mi = sum(
            self._segments[s].length_mi for s in segment_ids
            if s in self._segments and s not in seen_globally
        )

        # Build subgraph and solve CPP for this hike
        subgraph = nx.MultiGraph()
        for u, v, data in g.edges(data=True):
            if data['seg_id'] in segment_ids:
                if u not in subgraph:
                    subgraph.add_node(u, **g.nodes[u])
                if v not in subgraph:
                    subgraph.add_node(v, **g.nodes[v])
                subgraph.add_edge(u, v, **data)

        _, garbage_mi = self._solve_chinese_postman(subgraph)

        # Analyze shuttle benefit
        needs_shuttle, shuttle_savings, end_node, end_coord = self._analyze_shuttle_benefit(
            g, segment_ids, start_node
        )

        if needs_shuttle:
            garbage_mi = max(0, garbage_mi - shuttle_savings)
            end_name = f"Trailhead ({end_coord.lat:.4f}, {end_coord.lon:.4f})"
            shuttle_note = f"Get a ride from end back to start (saves {shuttle_savings:.1f} mi)"
        else:
            end_name = start_name
            end_coord = start_coord
            shuttle_note = ""

        # Compute actual walking order through segments
        ordered_segments = self._compute_walking_order(
            subgraph, segment_ids, start_node, end_node if needs_shuttle else start_node, seen_globally
        )
        segments_list = ordered_segments

        total_mi = challenge_mi + garbage_mi
        hours = total_mi / self.hiking_pace_mph

        return PlannedHike(
            hike_id=hike_id,
            category=self._categorize(hours),
            area_name=self._get_area_name(segment_ids),
            start_location=start_name,
            start_lat=start_coord.lat,
            start_lon=start_coord.lon,
            end_location=end_name,
            end_lat=end_coord.lat,
            end_lon=end_coord.lon,
            needs_shuttle=needs_shuttle,
            shuttle_savings_miles=round(shuttle_savings, 1),
            shuttle_note=shuttle_note,
            segments=segments_list,
            segment_ids=segment_ids.copy(),
            challenge_miles=round(challenge_mi, 2),
            garbage_miles=round(garbage_mi, 2),
            estimated_hours=round(hours, 1),
            hiking_pace_mph=self.hiking_pace_mph,
        )

    def _split_into_hikes(self, g: nx.MultiGraph, remaining: set[int]) -> list[set[int]]:
        """Split remaining segments into hike-sized groups.

        Uses geographic clustering with connectivity awareness.
        """
        if not remaining:
            return []

        # Build connectivity graph
        conn = nx.Graph()
        for seg_id in remaining:
            seg = self._segments.get(seg_id)
            if not seg:
                continue
            start = f"{seg.start_point.lat:.5f},{seg.start_point.lon:.5f}"
            end = f"{seg.end_point.lat:.5f},{seg.end_point.lon:.5f}"
            conn.add_edge(start, end, seg_id=seg_id, length=seg.length_mi)

        # Process each connected component
        all_groups = []
        for comp_nodes in nx.connected_components(conn):
            comp_segs = set()
            comp_total = 0
            for u, v, data in conn.edges(data=True):
                if u in comp_nodes and v in comp_nodes:
                    comp_segs.add(data['seg_id'])
                    comp_total += data['length']

            if comp_total <= self.MAX_CHALLENGE_MILES:
                all_groups.append(comp_segs)
            else:
                # Split large component
                splits = self._split_component(comp_segs)
                all_groups.extend(splits)

        # Merge small groups that are near each other
        merged = self._merge_small_groups(all_groups)
        return merged

    def _split_component(self, segment_ids: set[int]) -> list[set[int]]:
        """Split a large connected component into smaller chunks."""
        # Sort segments by latitude (north to south)
        seg_list = sorted(segment_ids, key=lambda s: (
            (self._segments[s].start_point.lat + self._segments[s].end_point.lat) / 2
            if s in self._segments else 0
        ))

        groups = []
        current = set()
        current_mi = 0

        for seg_id in seg_list:
            seg_mi = self._segments[seg_id].length_mi if seg_id in self._segments else 0

            if current_mi + seg_mi > self.TARGET_CHALLENGE_MILES and current:
                groups.append(current)
                current = set()
                current_mi = 0

            current.add(seg_id)
            current_mi += seg_mi

        if current:
            groups.append(current)

        return groups

    def _merge_small_groups(self, groups: list[set[int]]) -> list[set[int]]:
        """Merge groups that are too small and geographically close."""
        if not groups:
            return []

        MIN_CHALLENGE_MI = 4.0  # Minimum before trying to merge
        MAX_MERGE_DIST = 0.02   # ~1.4 miles in degrees

        result = []
        remaining = list(groups)

        while remaining:
            current = remaining.pop(0)
            current_mi = sum(
                self._segments[s].length_mi for s in current if s in self._segments
            )

            # Try to merge if too small
            while current_mi < MIN_CHALLENGE_MI and remaining:
                # Find centroid of current
                lats = [self._segments[s].start_point.lat for s in current if s in self._segments]
                lons = [self._segments[s].start_point.lon for s in current if s in self._segments]
                if not lats:
                    break
                curr_center = (sum(lats)/len(lats), sum(lons)/len(lons))

                # Find closest remaining group
                best_idx = -1
                best_dist = float('inf')

                for i, other in enumerate(remaining):
                    other_mi = sum(
                        self._segments[s].length_mi for s in other if s in self._segments
                    )
                    # Don't merge if combined would be too big
                    if current_mi + other_mi > self.MAX_CHALLENGE_MILES:
                        continue

                    o_lats = [self._segments[s].start_point.lat for s in other if s in self._segments]
                    o_lons = [self._segments[s].start_point.lon for s in other if s in self._segments]
                    if not o_lats:
                        continue
                    other_center = (sum(o_lats)/len(o_lats), sum(o_lons)/len(o_lons))

                    dist = ((curr_center[0] - other_center[0])**2 +
                           (curr_center[1] - other_center[1])**2) ** 0.5

                    if dist < MAX_MERGE_DIST and dist < best_dist:
                        best_dist = dist
                        best_idx = i

                if best_idx >= 0:
                    other = remaining.pop(best_idx)
                    current = current | other
                    current_mi = sum(
                        self._segments[s].length_mi for s in current if s in self._segments
                    )
                else:
                    break

            result.append(current)

        return result

    def optimize(self, completed_segments: set[int] | None = None) -> OptimizationResult:
        """Generate optimized hike plan with shuttle recommendations."""
        import time
        start_time = time.time()

        completed = completed_segments or set()
        remaining = set(self._segments.keys()) - completed

        if not remaining:
            return OptimizationResult(hikes=[])

        # Build graph of remaining segments
        t0 = time.time()
        g = nx.MultiGraph()
        for seg_id in remaining:
            seg = self._segments.get(seg_id)
            if not seg:
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
        print(f"  [1] Build graph: {len(g.nodes())} nodes, {len(g.edges())} edges ({(time.time()-t0)*1000:.1f}ms)")

        # Calculate theoretical minimum (Chinese Postman for full graph)
        t0 = time.time()
        odd_nodes = self._find_odd_degree_nodes(g)
        print(f"  [2] Found {len(odd_nodes)} odd-degree nodes (need matching)")
        total_challenge, min_garbage = self._solve_chinese_postman(g)
        print(f"  [3] Chinese Postman solution: {min_garbage:.1f} mi minimum garbage ({(time.time()-t0)*1000:.1f}ms)")

        # Split into hike-sized groups
        t0 = time.time()
        hike_groups = self._split_into_hikes(g, remaining)
        print(f"  [4] Split into {len(hike_groups)} hike groups ({(time.time()-t0)*1000:.1f}ms)")

        # Create hikes
        t0 = time.time()
        all_hikes = []
        seen_globally = completed.copy()

        for group in hike_groups:
            hike = self._create_hike(len(all_hikes) + 1, g, group, seen_globally)
            if hike:
                all_hikes.append(hike)
                seen_globally.update(group)
        print(f"  [5] Created {len(all_hikes)} hikes with shuttle analysis ({(time.time()-t0)*1000:.1f}ms)")

        # Sort by distance from home
        def hike_home_dist(h: PlannedHike) -> float:
            return haversine_distance(self.home_coord, Coordinate(lat=h.start_lat, lon=h.start_lon))

        all_hikes.sort(key=hike_home_dist)

        # Renumber
        for i, hike in enumerate(all_hikes):
            hike.hike_id = i + 1

        # Calculate totals from actual hikes (may differ from CPP due to splitting overhead)
        actual_garbage = sum(h.garbage_miles for h in all_hikes)
        shuttle_hikes = sum(1 for h in all_hikes if h.needs_shuttle)
        shuttle_savings = sum(h.shuttle_savings_miles for h in all_hikes)

        total_time = (time.time() - start_time) * 1000
        print(f"  [✓] Optimization complete: {total_challenge:.1f} challenge + {actual_garbage:.1f} garbage = {total_challenge + actual_garbage:.1f} mi")
        print(f"      {shuttle_hikes} shuttle hikes save {shuttle_savings:.1f} mi | Total time: {total_time:.0f}ms")

        return OptimizationResult(
            hikes=all_hikes,
            total_challenge_miles=round(total_challenge, 2),
            total_garbage_miles=round(actual_garbage, 2),
            total_hikes=len(all_hikes),
            short_hikes=sum(1 for h in all_hikes if h.category == HikeCategory.SHORT),
            medium_hikes=sum(1 for h in all_hikes if h.category == HikeCategory.MEDIUM),
            long_hikes=sum(1 for h in all_hikes if h.category == HikeCategory.LONG),
            shuttle_hikes=shuttle_hikes,
            total_shuttle_savings=round(shuttle_savings, 1),
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

        max_hours = min(available_hours, self.MAX_HIKE_HOURS)
        fitting_hikes = [h for h in result.hikes if h.estimated_hours <= max_hours]

        if fitting_hikes:
            return fitting_hikes[0]

        return min(result.hikes, key=lambda h: h.estimated_hours)


def create_optimizer_v6(
    trail_graph: TrailGraph,
    home_lat: float,
    home_lon: float,
    hiking_pace_mph: float = 2.5
) -> ShuttleAwareOptimizer:
    """Create a v6 shuttle-aware optimizer."""
    return ShuttleAwareOptimizer(trail_graph, home_lat, home_lon, hiking_pace_mph)
