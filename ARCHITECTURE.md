# Architecture

How a floor plan PDF becomes a walking route, what each stage owns, and where the data
formats are defined.

```
518 Archibus PDFs  (read-only source, ~/Downloads/UCalgary_Archibus_FloorPlans)
        |
        |  extract_floor.py        room polygons + room numbers, in page coordinates
        v
data/floors/<CODE>/<CODE>-<FLOOR>.json
        |
        |  georeference.py         fit each floor onto the real world
        |  floor_graph.py          rooms + corridors + doors -> a walkable graph
        |  outdoor.py              OpenStreetMap footpaths between buildings
        |  build_campus.py         runs all three, links floors vertically, exports
        v
out/navigation-graph.json      the whole campus, one routable graph
out/floors/<CODE>-<FLOOR>.geojson   room polygons per floor, WGS84 (opens in QGIS)
out/{georeference-fit,floor-graphs,vertical-links,wall-crossings}.json   build reports
        |
        |  app/index.html          search, Dijkstra, step-free mode, map
        v
a route
```

## Stages

### `pipeline/extract_floor.py` — PDF to room polygons
Owns: reading the PDFs, and nothing else. Knows about PDF internals; knows nothing about
the real world.

The room polygons live in a **linewidth-0 hairline layer** where every path is closed
(see the module docstring and BUILD-LOG for how that was found). Also filters out the
sheet border and title block, which appear identically in all 518 drawings.

Output is per floor, in PDF page points with y flipped to y-up:
```json
{"building_code":"BI","floor_id":"01","page":[842,595],
 "rooms":[{"number":"186","ring":[[x,y],...]}, ...],
 "unmatched_labels":[...], "stats":{...}}
```

### `pipeline/georeference.py` — page coordinates to the earth
Owns: the transform. Fits each floor's merged room outline to the building's ArcGIS
footprint, maximising IoU. Rotation is solved once per building; scale and position per
floor, because Archibus fits each floor to its own page. Floors that don't resemble the
footprint (roof structures, mezzanines) inherit the building's consensus scale rather
than being inflated to fill it.

Working CRS is **EPSG:3780** (Alberta 3TM) so distances are in metres; output is WGS84.

### `pipeline/floor_graph.py` — geometry to a walkable graph
Owns: what counts as a corridor, where a door is, and what connects to what. This is the
module that decides whether a route is true or merely plausible.

Nodes carry `pos` (metres, EPSG:3780), `kind` (`room` / `corridor` / `door`) and a room
`number`. Edges carry `weight` in metres, a `kind`, and a `geometry`.

Edge kinds, and what each one means physically:

| kind | meaning |
|---|---|
| `corridor` | along the corridor centreline |
| `room_to_door` | from inside a room to its doorway |
| `door_to_corridor` | from a doorway onto the corridor centreline |
| `room_to_room` | between two rooms through the wall they share (suites, ante-rooms) |
| `corridor_access` | a corridor space onto the centreline |
| `inferred_access` | fallback: a room that touches nothing, connected by a straight line |

### `pipeline/outdoor.py` — between buildings
Owns: the outdoor half. Real OSM pedestrian ways (`footway`, `path`, `steps`,
`pedestrian`, `living_street`) for the campus, trimmed to what's near a building we have
plans for, with degree-2 chains collapsed. Each building's entrance snaps to the nearest
path. `steps` are tagged so step-free routing can avoid them.

### `pipeline/build_campus.py` — assembly
Owns: everything that spans floors or buildings. Runs the three modules above, finds
vertical cores (a small space repeating in the same place on consecutive floors), picks
each building's entrance, and exports.

The exported graph:
```json
{"buildings": {"BI": {"name":"Biological Sciences","lon":..,"lat":..,"floors":[...]}},
 "nodes": [{"id":"BI|01|r12","lon":..,"lat":..,"kind":"room","building":"BI",
            "floor":"01","room":"186"}],
 "edges": [{"a":"BI|01|r12","b":"BI|01|d12","w":3.4,"kind":"room_to_door",
            "g":[[lon,lat],...]}]}
```
Node ids are `BUILDING|FLOOR|<type><n>`; `r`=room, `c`/`s`=corridor centreline,
`d`=door, `k`=corridor space, and outdoor nodes are `osm|x|y`.

### `pipeline/verify.py` and `pipeline/render_samples.py` — checking
`verify.py` reports reachability, vertical alignment and connectivity as numbers.
`render_samples.py` draws floors so they can be looked at — which is what actually caught
most of the real errors in this project.

### `app/index.html` — the viewer
Plain HTML and one script, no build step. Loads the graph, runs Dijkstra client-side
(a few milliseconds), and lazy-loads one floor's GeoJSON at a time to draw the plan under
the route. Step-free mode excludes `stairs`, `vertical_unknown` and `steps`.

## Running it

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python pipeline/extract_floor.py      # ~35 min, checkpoints per floor
./.venv/bin/python pipeline/build_campus.py       # ~20 min
./.venv/bin/python pipeline/verify.py
./.venv/bin/python pipeline/render_samples.py
python3 serve.py                                  # then http://localhost:8770/app/
```

Both long stages resume: `extract_floor.py` skips floors it has already written, and the
Overpass and ArcGIS responses are cached under `data/`.

## Coordinate systems, in one place

| where | system |
|---|---|
| inside a PDF / `data/floors/*.json` | page points, y-up, origin bottom-left |
| all geometry inside the pipeline | EPSG:3780, metres |
| everything exported | EPSG:4326 (WGS84) lon/lat |

Edge weights are always metres. A stair costs its horizontal distance plus a 20 m
equivalent-walking penalty per floor.
