#!/usr/bin/env python3
"""Test the v2 hike optimizer."""

import sys
sys.path.insert(0, ".")

from backend.services.btc_client import load_trail_data_from_cache
from backend.services.graph_builder import build_trail_graph
from backend.services.route_optimizer_v2 import create_optimizer_v2


def main():
    print("Loading trail data...")
    trail_data = load_trail_data_from_cache()

    print("Building trail graph...")
    graph = build_trail_graph(trail_data)

    # Use user's home location
    home_lat = 43.5835
    home_lon = -116.1428
    hiking_pace = 2.5  # mph

    print(f"\nHome: ({home_lat}, {home_lon})")
    print(f"Hiking pace: {hiking_pace} mph")

    print("\nCreating v2 optimizer...")
    optimizer = create_optimizer_v2(graph, home_lat, home_lon, hiking_pace)

    print("\nOptimizing full challenge...")
    result = optimizer.optimize()

    print(f"\n{'='*70}")
    print("FULL CHALLENGE OPTIMIZATION - V2")
    print(f"{'='*70}")
    print(f"\nSUMMARY:")
    print(f"  Total hikes: {result.total_hikes}")
    print(f"  - Short (1-2 hr): {result.short_hikes}")
    print(f"  - Medium (2-4 hr): {result.medium_hikes}")
    print(f"  - Long (4-6 hr): {result.long_hikes}")
    print(f"\n  Challenge miles: {result.total_challenge_miles}")
    print(f"  Garbage miles:   {result.total_garbage_miles}")
    print(f"  Total miles:     {result.total_miles}")
    print(f"  Efficiency:      {result.overall_efficiency:.1f}%")

    print(f"\n{'='*70}")
    print("HIKE DETAILS (showing first 10)")
    print(f"{'='*70}")

    for hike in result.hikes[:10]:
        print(f"\n--- Hike {hike.hike_id} ({hike.category.value.upper()}) ---")
        print(f"  Parking: {hike.parking_location}")
        print(f"  Returns to start: {'Yes' if hike.returns_to_start else 'No (shuttle needed)'}")
        print(f"  Challenge miles: {hike.challenge_miles:.1f}")
        print(f"  Garbage miles:   {hike.garbage_miles:.1f}")
        print(f"  Total miles:     {hike.total_miles:.1f}")
        print(f"  Time:            {hike.estimated_hours:.1f} hrs")
        print(f"  Efficiency:      {hike.efficiency:.0f}%")
        print(f"  Segments ({len(hike.segments)}):")
        for seg in hike.segments[:8]:
            marker = "✓" if seg.is_challenge else "↩"
            print(f"    {marker} {seg.seg_name} ({seg.length_mi:.2f} mi)")
        if len(hike.segments) > 8:
            print(f"    ... and {len(hike.segments) - 8} more")

    if len(result.hikes) > 10:
        print(f"\n... and {len(result.hikes) - 10} more hikes")

    print(f"\n{'='*70}")
    print("TIME BUDGET TEST: 3 hours available")
    print(f"{'='*70}")

    hike = optimizer.get_hike_for_time(3.0)
    if hike:
        print(f"\nRecommended hike ({hike.category.value}):")
        print(f"  Parking: {hike.parking_location}")
        print(f"  Challenge miles: {hike.challenge_miles:.1f}")
        print(f"  Garbage miles:   {hike.garbage_miles:.1f}")
        print(f"  Total miles:     {hike.total_miles:.1f}")
        print(f"  Time:            {hike.estimated_hours:.1f} hrs")
        print(f"  Efficiency:      {hike.efficiency:.0f}%")
        print(f"  Segments: {[s.seg_name for s in hike.segments if s.is_challenge][:5]}...")

    print("\n✓ V2 optimization test complete!")


if __name__ == "__main__":
    main()
