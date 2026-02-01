function app() {
    return {
        // State
        stats: null,
        route: null,
        graphData: null,
        timeBudget: {
            hours: 3,
            date: new Date().toISOString().split('T')[0],
            startTime: '09:00',
        },
        timeBudgetResult: null,
        selectedTrip: null,
        loading: false,
        map: null,
        trailLayer: null,
        tripLayer: null,

        // Initialize
        async init() {
            await this.loadStats();
            await this.loadGraphData();
            this.initMap();
        },

        // Load trail statistics
        async loadStats() {
            try {
                const res = await fetch('/api/trails/stats');
                this.stats = await res.json();
            } catch (e) {
                console.error('Failed to load stats:', e);
            }
        },

        // Load graph data for map
        async loadGraphData() {
            try {
                const res = await fetch('/api/trails/graph');
                this.graphData = await res.json();
            } catch (e) {
                console.error('Failed to load graph data:', e);
            }
        },

        // Initialize Leaflet map
        initMap() {
            // Center on Boise foothills
            this.map = L.map('map').setView([43.68, -116.15], 11);

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; OpenStreetMap contributors'
            }).addTo(this.map);

            // Draw trails if data loaded
            if (this.graphData) {
                this.drawTrails();
            }
        },

        // Draw all trails on map
        drawTrails() {
            if (this.trailLayer) {
                this.map.removeLayer(this.trailLayer);
            }

            const lines = [];
            for (const seg of this.graphData.segments) {
                const coords = seg.coordinates.map(c => [c[1], c[0]]); // [lat, lon]
                const line = L.polyline(coords, {
                    color: '#333',
                    weight: 2,
                    opacity: 0.6,
                });
                line.bindPopup(`<b>${seg.seg_name}</b><br>${seg.length_mi} mi`);
                lines.push(line);
            }

            this.trailLayer = L.layerGroup(lines);
            this.trailLayer.addTo(this.map);
        },

        // Highlight a trip on the map with numbered segments and direction arrows
        highlightTrip(trip) {
            if (!this.graphData || !trip) return;

            // Remove existing highlight
            if (this.tripLayer) {
                this.map.removeLayer(this.tripLayer);
            }

            // Build a map of seg_id to trip segment info (for ordering)
            const segMap = new Map();
            trip.segments.forEach((s, idx) => {
                segMap.set(s.seg_id, { index: idx + 1, isChallenge: s.direction === 'challenge' });
            });

            const layers = [];

            // Track used marker positions to offset overlapping numbers
            const usedPositions = new Map(); // key: "lat,lon" -> count of markers at that position

            for (const seg of this.graphData.segments) {
                const tripSeg = segMap.get(seg.seg_id);
                if (tripSeg) {
                    const coords = seg.coordinates.map(c => [c[1], c[0]]);
                    const color = tripSeg.isChallenge ? '#2d5a27' : '#996600';

                    // Draw the segment line
                    const line = L.polyline(coords, {
                        color: color,
                        weight: 4,
                        opacity: 1,
                    });
                    line.bindPopup(`<b>#${tripSeg.index}: ${seg.seg_name}</b><br>${seg.length_mi.toFixed(1)} mi`);
                    layers.push(line);

                    // Add direction arrow at midpoint
                    if (coords.length >= 2) {
                        const midIdx = Math.floor(coords.length / 2);
                        const midPoint = coords[midIdx];
                        const nextPoint = coords[Math.min(midIdx + 1, coords.length - 1)];

                        // Calculate angle for arrow
                        const angle = Math.atan2(
                            nextPoint[0] - midPoint[0],
                            nextPoint[1] - midPoint[1]
                        ) * 180 / Math.PI;

                        // Arrow marker
                        const arrowIcon = L.divIcon({
                            html: `<div style="transform: rotate(${angle}deg); color: ${color}; font-size: 16px; font-weight: bold;">▲</div>`,
                            className: 'arrow-icon',
                            iconSize: [20, 20],
                            iconAnchor: [10, 10],
                        });
                        const arrow = L.marker(midPoint, { icon: arrowIcon });
                        layers.push(arrow);
                    }

                    // Add segment number label at start, with offset for overlapping markers
                    const posKey = `${coords[0][0].toFixed(5)},${coords[0][1].toFixed(5)}`;
                    const posCount = usedPositions.get(posKey) || 0;
                    usedPositions.set(posKey, posCount + 1);

                    // Offset overlapping markers in a circle pattern
                    let markerLat = coords[0][0];
                    let markerLon = coords[0][1];
                    if (posCount > 0) {
                        const offsetAngle = (posCount * 90) * Math.PI / 180; // 90 degree increments
                        const offsetDist = 0.0003; // ~30 meters
                        markerLat += Math.cos(offsetAngle) * offsetDist;
                        markerLon += Math.sin(offsetAngle) * offsetDist;
                    }

                    const numIcon = L.divIcon({
                        html: `<div class="seg-number" style="background: ${color};">${tripSeg.index}</div>`,
                        className: 'seg-number-icon',
                        iconSize: [28, 28],
                        iconAnchor: [14, 14],
                    });
                    const numMarker = L.marker([markerLat, markerLon], { icon: numIcon, zIndexOffset: 1000 + tripSeg.index });
                    numMarker.bindPopup(`<b>#${tripSeg.index}: ${seg.seg_name}</b><br>${seg.length_mi.toFixed(1)} mi`);
                    layers.push(numMarker);
                }
            }

            // Add start marker (parking/trailhead)
            if (trip.trailhead_lat && trip.trailhead_lon) {
                const startIcon = L.divIcon({
                    html: '<div class="start-marker">P</div>',
                    className: 'start-marker-icon',
                    iconSize: [28, 28],
                    iconAnchor: [14, 14],
                });
                const startMarker = L.marker([trip.trailhead_lat, trip.trailhead_lon], { icon: startIcon });
                startMarker.bindPopup(`<b>Start: ${trip.area_name}</b><br>Park here`);
                layers.push(startMarker);
            }

            // Add end marker for shuttle hikes
            if (trip.needs_shuttle && trip.end_lat && trip.end_lon) {
                const endIcon = L.divIcon({
                    html: '<div class="end-marker">🚗</div>',
                    className: 'end-marker-icon',
                    iconSize: [28, 28],
                    iconAnchor: [14, 14],
                });
                const endMarker = L.marker([trip.end_lat, trip.end_lon], { icon: endIcon });
                endMarker.bindPopup(`<b>End: Pickup point</b><br>${trip.shuttle_note || 'Get shuttle here'}`);
                layers.push(endMarker);
            }

            this.tripLayer = L.layerGroup(layers);
            this.tripLayer.addTo(this.map);

            // Fit bounds to trip
            const polylines = layers.filter(l => l.getBounds);
            if (polylines.length > 0) {
                const bounds = L.latLngBounds(
                    polylines.flatMap(l => [
                        l.getBounds().getSouthWest(),
                        l.getBounds().getNorthEast()
                    ])
                );
                if (bounds.isValid()) {
                    this.map.fitBounds(bounds, { padding: [50, 50] });
                }
            }
        },

        // Optimize full challenge
        async optimizeRoute() {
            this.loading = true;
            try {
                const res = await fetch('/api/optimize');
                this.route = await res.json();
            } catch (e) {
                console.error('Failed to optimize:', e);
            } finally {
                this.loading = false;
            }
        },

        // Get trip for time budget
        async getTimeBudgetTrip() {
            this.loading = true;
            try {
                const res = await fetch('/api/time-budget', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        available_hours: parseFloat(this.timeBudget.hours),
                        target_date: this.timeBudget.date,
                        start_time: this.timeBudget.startTime,
                        prefer_close: true,
                    }),
                });
                this.timeBudgetResult = await res.json();

                // Highlight on map
                if (this.timeBudgetResult?.trip) {
                    this.highlightTrip(this.timeBudgetResult.trip);
                }
            } catch (e) {
                console.error('Failed to get time budget trip:', e);
            } finally {
                this.loading = false;
            }
        },

        // Select a trip from the list
        selectTrip(trip) {
            if (this.selectedTrip?.trip_id === trip.trip_id) {
                // Toggle off if already selected
                this.selectedTrip = null;
                if (this.tripLayer) {
                    this.map.removeLayer(this.tripLayer);
                    this.tripLayer = null;
                }
            } else {
                this.selectedTrip = trip;
                this.highlightTrip(trip);
            }
        },

        // Open Google Maps directions to trailhead
        openDirections(trip) {
            if (!trip.trailhead_lat || !trip.trailhead_lon) return;
            const url = `https://www.google.com/maps/dir/?api=1&destination=${trip.trailhead_lat},${trip.trailhead_lon}`;
            window.open(url, '_blank');
        },

        // Show trip on map and scroll to it
        showOnMap(trip) {
            this.highlightTrip(trip);
            // Scroll map into view
            document.getElementById('map').scrollIntoView({ behavior: 'smooth' });
        },
    };
}
