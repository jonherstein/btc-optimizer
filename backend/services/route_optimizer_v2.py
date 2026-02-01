"""Route optimization v2 - focused on practical hiking logistics.

Key principles:
1. Every hike must return to your car (loop or out-and-back)
2. Minimize garbage miles (non-challenge connector/backtracking)
3. Categorize hikes: Short (1-2hr), Medium (2-4hr), Long (4-6hr)
4. Never exceed 6 hours of hiking
"""

from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from backend.models.trail import Coordinate
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
    is_challenge: bool  # True = counts toward challenge, False = garbage miles


@dataclass
class PlannedHike:
    """A complete hike plan with car logistics."""
    hike_id: int
    category: HikeCategory
    area_name: str

    # Car logistics
    parking_location: str
    parking_lat: float
    parking_lon: float
    returns_to_start: bool = True

    # Route details
    segments: list[HikeSegment] = field(default_factory=list)
    segment_ids: set[int] = field(default_factory=set)

    # Miles breakdown
    challenge_miles: float = 0.0
    garbage_miles: float = 0.0

    # Time (calculated from miles / pace)
    estimated_hours: float = 0.0
    hiking_pace_mph: float = 2.5  # Store pace for time calculations

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


class HikeOptimizer:
    """Optimizes hikes with practical car logistics."""

    MAX_HIKE_HOURS = 6.0
    MAX_CHALLENGE_MILES_PER_HIKE = 8.0  # More conservative to stay under 6 hrs with garbage

    def __init__(self, trail_graph: TrailGraph, home_lat: float, home_lon: float, hiking_pace_mph: float = 2.5):
        self.graph = trail_graph
        self.home_lat = home_lat
        self.home_lon = home_lon
        self.home_coord = Coordinate(lat=home_lat, lon=home_lon)
        self.hiking_pace_mph = hiking_pace_mph

        # Precompute segment data
        self._segments = {s.seg_id: s for s in self.graph.trail_data.segments}
        self._segment_centers = {}
        for seg in self.graph.trail_data.segments:
            mid_idx = len(seg.coordinates) // 2
            self._segment_centers[seg.seg_id] = seg.coordinates[mid_idx]

    def _get_area_name(self, segment_ids: set[int]) -> str:
        """Generate area name from trail names."""
        names = set()
        for seg_id in segment_ids:
            seg = self._segments.get(seg_id)
            if seg:
                # Extract base name (remove trailing number)
                parts = seg.seg_name.rsplit(' ', 1)
                base = parts[0] if len(parts) > 1 and parts[1].isdigit() else seg.seg_name
                names.add(base)
        sorted_names = sorted(names)[:3]
        return " / ".join(sorted_names) if sorted_names else "Trail Area"

    def _find_parking(self, segment_ids: set[int]) -> tuple[float, float, str]:
        """Find best parking spot for a set of segments (closest to home)."""
        endpoints = []
        for seg_id in segment_ids:
            seg = self._segments.get(seg_id)
            if seg:
                endpoints.append(seg.start_point)
                endpoints.append(seg.end_point)

        if not endpoints:
            return self.home_lat, self.home_lon, "Unknown"

        best = min(endpoints, key=lambda c: haversine_distance(self.home_coord, c))
        return best.lat, best.lon, f"Trailhead ({best.lat:.4f}, {best.lon:.4f})"

    def _estimate_garbage_miles(self, segment_ids: set[int]) -> float:
        """Estimate garbage miles based on trail connectivity.

        - Well-connected segments (loops possible): ~30% overhead
        - Sparse/linear segments: ~80% overhead (out-and-back)
        """
        if not segment_ids:
            return 0

        # Check connectivity by counting endpoint matches
        endpoints = defaultdict(int)
        for seg_id in segment_ids:
            seg = self._segments.get(seg_id)
            if seg:
                start_key = f"{seg.start_point.lat:.4f},{seg.start_point.lon:.4f}"
                end_key = f"{seg.end_point.lat:.4f},{seg.end_point.lon:.4f}"
                endpoints[start_key] += 1
                endpoints[end_key] += 1

        # Count "dead ends" (endpoints that appear only once)
        dead_ends = sum(1 for v in endpoints.values() if v == 1)
        total_endpoints = len(endpoints)

        # More dead ends = more backtracking needed
        if total_endpoints == 0:
            return 0

        dead_end_ratio = dead_ends / total_endpoints

        # Calculate total challenge miles
        challenge_miles = sum(self._segments[s].length_mi for s in segment_ids if s in self._segments)

        # Estimate garbage based on connectivity
        if dead_end_ratio < 0.3:
            # Good connectivity - mostly loops
            return challenge_miles * 0.25
        elif dead_end_ratio < 0.5:
            # Moderate connectivity
            return challenge_miles * 0.50
        else:
            # Poor connectivity - lots of out-and-back
            return challenge_miles * 0.80

    def _categorize(self, hours: float) -> HikeCategory:
        if hours <= 2:
            return HikeCategory.SHORT
        elif hours <= 4:
            return HikeCategory.MEDIUM
        else:
            return HikeCategory.LONG

    def _cluster_segments(self, segment_ids: set[int], radius_mi: float = 2.5) -> list[set[int]]:
        """Cluster segments by geographic proximity."""
        remaining = segment_ids.copy()
        clusters = []
        radius_m = radius_mi * 1609.34

        while remaining:
            # Start from segment closest to home
            seed = min(remaining, key=lambda s: haversine_distance(self.home_coord, self._segment_centers[s]))
            seed_center = self._segment_centers[seed]

            cluster = {seed}
            remaining.remove(seed)

            # Add all segments within radius
            for seg_id in list(remaining):
                if haversine_distance(seed_center, self._segment_centers[seg_id]) <= radius_m:
                    cluster.add(seg_id)
                    remaining.remove(seg_id)

            clusters.append(cluster)

        return clusters

    def _create_hike(self, hike_id: int, segment_ids: set[int]) -> PlannedHike:
        """Create a hike from a set of segments."""
        challenge_miles = sum(self._segments[s].length_mi for s in segment_ids if s in self._segments)
        garbage_miles = self._estimate_garbage_miles(segment_ids)
        total_miles = challenge_miles + garbage_miles
        hours = total_miles / self.hiking_pace_mph

        parking_lat, parking_lon, parking_desc = self._find_parking(segment_ids)

        segments = []
        for seg_id in sorted(segment_ids, key=lambda s: self._segments[s].seg_name if s in self._segments else ""):
            seg = self._segments.get(seg_id)
            if seg:
                segments.append(HikeSegment(
                    seg_id=seg_id,
                    seg_name=seg.seg_name,
                    length_mi=round(seg.length_mi, 2),
                    is_challenge=True
                ))

        # Add garbage miles note
        if garbage_miles > 0:
            segments.append(HikeSegment(
                seg_id=0,
                seg_name="Return/connectors (estimated)",
                length_mi=round(garbage_miles, 2),
                is_challenge=False
            ))

        return PlannedHike(
            hike_id=hike_id,
            category=self._categorize(hours),
            area_name=self._get_area_name(segment_ids),
            parking_location=parking_desc,
            parking_lat=parking_lat,
            parking_lon=parking_lon,
            segments=segments,
            segment_ids=segment_ids,
            challenge_miles=round(challenge_miles, 2),
            garbage_miles=round(garbage_miles, 2),
            estimated_hours=round(hours, 1),
            hiking_pace_mph=self.hiking_pace_mph,
        )

    def _split_large_cluster(self, segment_ids: set[int]) -> list[set[int]]:
        """Split a cluster into hikes of appropriate size."""
        challenge_miles = sum(self._segments[s].length_mi for s in segment_ids if s in self._segments)
        garbage_miles = self._estimate_garbage_miles(segment_ids)
        total_hours = (challenge_miles + garbage_miles) / self.hiking_pace_mph

        # Check both challenge miles and total time
        if challenge_miles <= self.MAX_CHALLENGE_MILES_PER_HIKE and total_hours <= self.MAX_HIKE_HOURS:
            return [segment_ids]

        # Need to split - group by sub-clusters (tighter radius)
        sub_clusters = self._cluster_segments(segment_ids, radius_mi=0.75)

        # Merge small sub-clusters, checking both miles and time
        result = []
        current = set()

        for cluster in sub_clusters:
            test_combined = current | cluster
            combined_challenge = sum(self._segments[s].length_mi for s in test_combined if s in self._segments)
            combined_garbage = self._estimate_garbage_miles(test_combined)
            combined_hours = (combined_challenge + combined_garbage) / self.hiking_pace_mph

            if combined_challenge <= self.MAX_CHALLENGE_MILES_PER_HIKE and combined_hours <= self.MAX_HIKE_HOURS:
                current = test_combined
            else:
                if current:
                    result.append(current)
                current = cluster

        if current:
            result.append(current)

        # Final check - split any remaining over-time clusters by halving
        final_result = []
        for cluster in result:
            challenge = sum(self._segments[s].length_mi for s in cluster if s in self._segments)
            garbage = self._estimate_garbage_miles(cluster)
            hours = (challenge + garbage) / self.hiking_pace_mph

            if hours > self.MAX_HIKE_HOURS and len(cluster) > 1:
                # Split in half
                sorted_segs = sorted(cluster, key=lambda s: self._segment_centers[s].lat)
                mid = len(sorted_segs) // 2
                final_result.append(set(sorted_segs[:mid]))
                final_result.append(set(sorted_segs[mid:]))
            else:
                final_result.append(cluster)

        return final_result

    def optimize(self, completed_segments: set[int] | None = None) -> OptimizationResult:
        """Generate optimized hike plan for the full challenge."""
        completed = completed_segments or set()
        remaining = set(self._segments.keys()) - completed

        if not remaining:
            return OptimizationResult(hikes=[])

        # Cluster by geography
        clusters = self._cluster_segments(remaining, radius_mi=2.5)

        # Split large clusters and create hikes
        hikes = []
        hike_id = 1

        for cluster in clusters:
            sub_hikes = self._split_large_cluster(cluster)
            for sub in sub_hikes:
                hike = self._create_hike(hike_id, sub)
                hikes.append(hike)
                hike_id += 1

        # Sort by distance from home
        hikes.sort(key=lambda h: haversine_distance(self.home_coord, Coordinate(lat=h.parking_lat, lon=h.parking_lon)))

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

    def get_hike_for_time(self, available_hours: float, completed_segments: set[int] | None = None) -> PlannedHike | None:
        """Get a single hike that fits within available time."""
        completed = completed_segments or set()
        remaining = set(self._segments.keys()) - completed

        if not remaining:
            return None

        max_hours = min(available_hours, self.MAX_HIKE_HOURS)
        # Conservative estimate: assume 60% overhead for garbage miles
        max_challenge_mi = (max_hours * self.hiking_pace_mph) / 1.6

        # Find closest cluster
        clusters = self._cluster_segments(remaining, radius_mi=2.5)

        if not clusters:
            return None

        # Take from the closest cluster
        cluster = clusters[0]

        # Select segments that fit, smaller segments first for better packing
        sorted_segs = sorted(cluster, key=lambda s: self._segments[s].length_mi)
        selected = set()
        selected_mi = 0

        for seg_id in sorted_segs:
            seg_mi = self._segments[seg_id].length_mi
            if selected_mi + seg_mi <= max_challenge_mi:
                selected.add(seg_id)
                selected_mi += seg_mi

        if not selected and sorted_segs:
            # At minimum take one segment
            selected.add(sorted_segs[0])

        hike = self._create_hike(1, selected)

        # Verify it fits - if not, reduce segments
        while hike.estimated_hours > max_hours and len(selected) > 1:
            # Remove the largest segment
            largest = max(selected, key=lambda s: self._segments[s].length_mi)
            selected.remove(largest)
            hike = self._create_hike(1, selected)

        return hike


def create_optimizer_v2(trail_graph: TrailGraph, home_lat: float, home_lon: float, hiking_pace_mph: float = 2.5) -> HikeOptimizer:
    """Create a v2 hike optimizer."""
    return HikeOptimizer(trail_graph, home_lat, home_lon, hiking_pace_mph)
