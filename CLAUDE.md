# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
pip3 install -r requirements.txt  # Install dependencies
python3 -m uvicorn backend.main:app --reload --port 8003  # Start dev server
python3 test_optimizer.py         # Test route optimization
```

## Architecture

**Python FastAPI application** that optimizes hiking routes for the Boise Trails Challenge.

```
backend/
├── main.py                   # FastAPI app, serves templates
├── config.py                 # Settings (home: 3410 E Vortex Dr)
├── models/trail.py           # Trail/Segment Pydantic models
├── services/
│   ├── btc_client.py         # Loads trail data from cache/BTC API
│   ├── graph_builder.py      # Builds networkx graph from geometry
│   ├── route_optimizer_v2.py # Main optimizer (geographic clustering)
│   ├── drive_time.py         # OpenRouteService for drive times
│   └── daylight.py           # Sunset calculations
└── api/
    ├── routes.py             # API endpoints
    └── schemas.py            # Request/response schemas

frontend/
├── templates/                # Jinja2 + htmx (index.html, settings.html)
└── static/                   # CSS + Alpine.js + Leaflet map

data/
└── trail_cache.json          # 245 segments, 164.7 challenge miles
```

## Key Concepts

- **Challenge miles**: Segments that count toward BTC completion
- **Garbage miles**: Return paths, connectors, backtracking (don't count but must be hiked)
- **Efficiency**: challenge_miles / total_miles (target: 60-80%)
- **Hike categories**: Short (1-2hr), Medium (2-4hr), Long (4-6hr), max 6hr
- **Connectivity**: Segments with many dead-ends = more garbage miles (out-and-back)

## API Endpoints

- `GET /api/trails/stats` - Trail statistics
- `GET /api/trails/graph` - GeoJSON for map
- `GET /api/optimize` - Full challenge optimization (31 hikes)
- `POST /api/time-budget` - "I have X hours" recommendation
- `GET/PUT /api/settings` - User home location, pace
