"""
Assemble every floor of every building into one routable campus graph.

Run:  .venv/bin/python pipeline/build_campus.py

What this stage owns
--------------------
The earlier stages produce one floor at a time. This one joins them up:

  * georeferences each building's floors (georeference.py)
  * builds each floor's room/door/corridor graph (floor_graph.py)
  * links floors vertically through stair cores
  * links buildings to each other across the outdoors using real OpenStreetMap
    pedestrian paths
  * writes out/navigation-graph.json plus a per-floor GeoJSON so any stage's
    output can be opened in QGIS and inspected directly

Vertical links, and why they aren't read from a layer
-----------------------------------------------------
There is no stairs layer in these PDFs. But a stair core is geometrically
obvious once floors are georeferenced: it is a small space occupying *the same
footprint on consecutive floors*. So vertical links are found by overlaying
each pair of adjacent floors and looking for small spaces that coincide.

Room numbering then says what kind of core it is. This dataset's suffixes were
worked out from the data rather than assumed:

    Z  corridor   (614 of them; confirmed by the previous pipeline too)
    Y  stairs     (252; confirmed by the existence of "YUP" and "YDN" numbers,
                   the conventional CAD markers for a stair going up/down)
    V  shafts     (819, median area 1.8 m2 - ducts, far too small to walk)
    W  washrooms  (505, median 9.2 m2)

**Elevators cannot be identified from this data.** No suffix corresponds to
them and there is no room-type table to consult. That matters for accessible
routing, so the stance here is deliberately conservative: a vertical link is
tagged `stairs` when the core is Y-numbered, and `vertical_unknown` otherwise.
Accessible routing excludes both. The consequence is that step-free routing
will refuse some journeys that are actually possible by lift, which is the
right way round - the opposite error would send a wheelchair user to a
staircase.

Costs
-----
Edge weights are metres, so a route's length is a real distance. Climbing a
floor is charged an extra STAIR_FLOOR_PENALTY_M of equivalent walking, because
a flight of stairs costs more effort and time than the couple of metres of
horizontal travel it represents.
"""
import glob
import gzip
import json
import math
import os
import re
import shutil
import sys
import time

import networkx as nx
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import floor_graph as fg
from georeference import (BUILDING_CODE_ALIASES, apply_transform, fetch_footprints,
                          floor_outline, rings_to_polygon, solve_building, to_wgs84)
import outdoor

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOOR_DIR = os.path.join(PROJECT, "data", "floors")
OUT_DIR = os.path.join(PROJECT, "out")

# A stair core is a small space sitting in the same place on two floors.
VERTICAL_MAX_AREA_M2 = 90.0
VERTICAL_MIN_IOU = 0.55
STAIR_FLOOR_PENALTY_M = 20.0

# Stairs are numbered with a Y suffix in this dataset; confirmed by the
# existence of "YUP"/"YDN" numbers, the CAD convention for a stair's up and
# down runs.
STAIR_NUMBER_RE = re.compile(r"^\d+Y", re.IGNORECASE)


def floor_sort_key(floor_id):
    """B2 < B1 < G1 < 01 < 01M < 02 ... so 'consecutive floors' means something."""
    s = str(floor_id).upper()
    if s.startswith("B"):
        try:
            return -int(s[1:] or 1)
        except ValueError:
            return -1
    if s.startswith("G"):
        return 0
    if s.startswith("P"):          # parkade levels sit below ground
        try:
            return -10 - int(s[1:] or 1)
        except ValueError:
            return -10
    digits = ""
    for ch in s:
        if ch.isdigit():
            digits += ch
        else:
            break
    if digits:
        base = int(digits)
        rest = s[len(digits):]
        # 01M (mezzanine) sits just above 01; 08RF is the top of 08
        return base + (0.5 if rest.startswith("M") else 0.0)
    return 999


def load_building(code):
    files = sorted(glob.glob(os.path.join(FLOOR_DIR, code, "*.json")))
    floors, data = [], {}
    for path in files:
        with open(path) as fh:
            d = json.load(fh)
        outline = floor_outline(d)
        if outline is None or outline.area <= 0:
            data[d["floor_id"]] = d          # keep, but it can't be fitted
            continue
        floors.append((d["floor_id"], outline))
        data[d["floor_id"]] = d
    return floors, data


def placed_rooms(floor_data, params):
    rooms = []
    for room in floor_data.get("rooms", []):
        ring = room.get("ring")
        if not ring or len(ring) < 4:
            continue
        try:
            g = apply_transform(Polygon(ring), params)
            if g.is_valid and not g.is_empty and g.area > 0.5:
                rooms.append({"number": room.get("number"), "poly": g})
        except Exception:
            continue
    return rooms


def link_floors(graph, code, floor_rooms, floor_order):
    """
    Join consecutive floors wherever a small space repeats in the same place.

    Returns a list of link records for the verification report.
    """
    links = []
    for lower, upper in zip(floor_order, floor_order[1:]):
        for ni, (num_a, poly_a, node_a) in enumerate(floor_rooms.get(lower, [])):
            if poly_a.area > VERTICAL_MAX_AREA_M2:
                continue
            for (num_b, poly_b, node_b) in floor_rooms.get(upper, []):
                if poly_b.area > VERTICAL_MAX_AREA_M2:
                    continue
                inter = poly_a.intersection(poly_b).area
                if inter <= 0:
                    continue
                iou = inter / (poly_a.area + poly_b.area - inter)
                if iou < VERTICAL_MIN_IOU:
                    continue
                if not (graph.has_node(node_a) and graph.has_node(node_b)):
                    continue

                num = (num_a or num_b or "").upper().replace(" ", "")
                # digits then a Y suffix: "142Y", "142YDN". The first attempt
                # at this used rstrip() and matched almost nothing - only 200
                # of 4,031 links came out as stairs.
                kind = "stairs" if STAIR_NUMBER_RE.match(num) else "vertical_unknown"

                ax, ay = graph.nodes[node_a]["pos"]
                bx, by = graph.nodes[node_b]["pos"]
                horizontal = math.dist((ax, ay), (bx, by))
                graph.add_edge(node_a, node_b,
                               weight=STAIR_FLOOR_PENALTY_M + horizontal,
                               kind=kind, geometry=None)
                links.append({"building": code, "from": lower, "to": upper,
                              "number": num_a or num_b, "kind": kind,
                              "iou": round(iou, 3),
                              "horizontal_offset_m": round(horizontal, 2)})
                break
    return links


def pick_entrance(graph, code, footprint, floor_rooms, floor_order):
    """
    Where the building meets the outdoors.

    No exterior-door data exists, so the entrance stands in as a corridor node
    near the building's outer wall on its lowest well-connected floor. That
    puts the indoor/outdoor handover at the edge of the building rather than
    in the middle of it.

    The important subtlety is **which part of the building it attaches to**.
    The first version took the lowest floor outright, and that quietly broke
    cross-building routing: Biological Sciences' lowest floor is a 22-node
    basement fragment that isn't connected to the other 2,934 nodes of the
    building, so the entire building hung off the campus network by a stub and
    only 7 of its 1,036 rooms could be reached from outside. Campus-wide that
    left 13% of rooms routable.

    So the entrance is chosen from the building's **largest connected
    component** - the part that actually is the building - and only then by
    floor and proximity to the outer wall.
    """
    own = [n for n, d in graph.nodes(data=True) if d.get("building") == code]
    if not own:
        return None
    sub = graph.subgraph(own)
    try:
        main = max(nx.connected_components(sub), key=len)
    except ValueError:
        return None

    boundary = footprint.exterior if footprint.geom_type == "Polygon" else \
        unary_union([g.exterior for g in footprint.geoms])

    candidates = [(n, graph.nodes[n]) for n in main
                  if graph.nodes[n].get("kind") == "corridor"]
    if not candidates:
        candidates = [(n, graph.nodes[n]) for n in main]
    if not candidates:
        return None

    # prefer the lowest floor that is actually part of the main component
    floors_present = {d.get("floor") for _, d in candidates}
    lowest = min(floors_present, key=floor_sort_key)
    on_lowest = [(n, d) for n, d in candidates if d.get("floor") == lowest] or candidates
    return min(on_lowest, key=lambda nd: boundary.distance(Point(nd[1]["pos"])))[0]


def main():
    started = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    footprints = fetch_footprints(os.path.join(PROJECT, "data", "footprints.json"))

    campus = nx.Graph()
    buildings, fit_report, vertical_report, floor_report = {}, [], [], []
    wall_crossings = []
    entrances = {}

    codes = sorted(os.path.basename(p) for p in glob.glob(os.path.join(FLOOR_DIR, "*")))
    for code in codes:
        fp_key = BUILDING_CODE_ALIASES.get(code, code)
        if fp_key not in footprints:
            print(f"[skip] {code}: no ArcGIS footprint")
            continue
        footprint = rings_to_polygon(footprints[fp_key]["rings"])
        floors, data = load_building(code)
        if not floors:
            print(f"[skip] {code}: no floor had usable room geometry")
            continue

        rotation, ref_floor, fits = solve_building(floors, footprint)
        floor_order = sorted([fid for fid, _ in floors], key=floor_sort_key)
        floor_rooms = {}

        for floor_id, _ in floors:
            params = fits[floor_id]
            rooms = placed_rooms(data[floor_id], params)
            graph, stats, spaces = fg.build(code, floor_id, rooms)

            # the central correctness check, done here because it needs the
            # room polygons, which only exist at this point in the pipeline
            circulation = unary_union(
                [poly for node_id, poly in spaces
                 if graph.nodes[node_id].get("is_corridor")]) if spaces else None
            write_floor_geojson(code, floor_id, spaces, graph)
            for bad in fg.edges_through_walls(graph, spaces, circulation):
                wall_crossings.append({"building": code, "floor": floor_id, **bad})
            stats["wall_crossings"] = sum(
                1 for w in wall_crossings
                if w["building"] == code and w["floor"] == floor_id)

            for n, d in graph.nodes(data=True):
                campus.add_node(n, building=code, floor=floor_id, **d)
            for u, v, d in graph.edges(data=True):
                campus.add_edge(u, v, **d)

            floor_rooms[floor_id] = [
                (graph.nodes[n].get("number"), poly, n) for n, poly in spaces]

            fit_report.append({
                "building": code, "floor": floor_id,
                "iou": round(params["score"], 3),
                "coverage": round(params["coverage"], 3),
                "scale_m_per_pt": round(params["scale"], 4),
                "rotation": round(rotation, 2),
                "trusted_scale": bool(params.get("trusted", True)),
                "scale_source": params.get("scale_source", "own fit"),
            })
            floor_report.append({"building": code, "floor": floor_id, **stats})

        vertical_report.extend(link_floors(campus, code, floor_rooms, floor_order))

        entrance = pick_entrance(campus, code, footprint, floor_rooms, floor_order)
        if entrance:
            entrances[code] = entrance
            campus.nodes[entrance]["entrance"] = True

        centroid = footprint.centroid
        lon, lat = to_wgs84(centroid.x, centroid.y)
        buildings[code] = {
            "name": footprints[fp_key].get("name") or code,
            "lon": round(lon, 6), "lat": round(lat, 6),
            "floors": floor_order,
        }
        rooms_here = sum(1 for n, d in campus.nodes(data=True)
                         if d.get("building") == code and d.get("number"))
        print(f"  {code:5s} floors={len(floors):3d} rotation={rotation:6.1f} "
              f"numbered_rooms={rooms_here:5d}")

    # ---- outdoors
    outdoor_stats = outdoor.connect(campus, buildings, entrances,
                                    cache_dir=os.path.join(PROJECT, "data"))

    export(campus, buildings, fit_report, floor_report, vertical_report,
           outdoor_stats, wall_crossings)
    print(f"\ndone in {time.time()-started:.0f}s")


def write_floor_geojson(code, floor_id, spaces, graph):
    """
    One GeoJSON per floor, in plain WGS84.

    Two reasons this exists rather than everything living in the one graph
    file: the viewer only ever draws one floor at a time, so shipping every
    room polygon on campus up front would be wasteful; and a GeoJSON per floor
    can be dragged straight into QGIS to see what a stage actually produced.
    """
    features = []
    for node_id, poly in spaces:
        data = graph.nodes[node_id]
        try:
            ring = [[round(c, 6) for c in to_wgs84(x, y)]
                    for x, y in poly.exterior.coords]
        except Exception:
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "node": node_id,
                "room": data.get("number"),
                "corridor": bool(data.get("is_corridor")),
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    target_dir = os.path.join(OUT_DIR, "floors")
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, f"{code}-{floor_id}.geojson"), "w") as fh:
        json.dump({"type": "FeatureCollection", "features": features}, fh)


def export(graph, buildings, fit_report, floor_report, vertical_report,
           outdoor_stats, wall_crossings):
    nodes = []
    for n, d in graph.nodes(data=True):
        lon, lat = to_wgs84(*d["pos"])
        node = {"id": n, "lon": round(lon, 6), "lat": round(lat, 6),
                "kind": d.get("kind"), "building": d.get("building"),
                "floor": d.get("floor")}
        if d.get("number"):
            node["room"] = d["number"]
        for flag in ("entrance", "door_inferred", "reaches_corridor_via_room", "is_corridor"):
            if d.get(flag):
                node[flag] = True
        nodes.append(node)

    edges = []
    for u, v, d in graph.edges(data=True):
        edge = {"a": u, "b": v, "w": round(float(d.get("weight", 1.0)), 2),
                "kind": d.get("kind", "walk")}
        geom = d.get("geometry")
        if geom is not None and geom.length > 0 and len(geom.coords) > 2:
            edge["g"] = [[round(c, 6) for c in to_wgs84(x, y)]
                         for x, y in geom.simplify(0.4).coords]
        edges.append(edge)

    payload = {"buildings": buildings, "nodes": nodes, "edges": edges}
    path = os.path.join(OUT_DIR, "navigation-graph.json")
    with open(path, "w") as fh:
        json.dump(payload, fh)
    with open(path, "rb") as src, gzip.open(path + ".gz", "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)

    for name, report in (("georeference-fit", fit_report),
                         ("floor-graphs", floor_report),
                         ("vertical-links", vertical_report),
                         ("wall-crossings", wall_crossings)):
        with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as fh:
            json.dump(report, fh, indent=1)

    rooms = sum(1 for n in nodes if n.get("room"))
    comps = nx.number_connected_components(graph)
    print(f"\n{'='*66}")
    print(f"{len(buildings)} buildings | {len(nodes)} nodes | {len(edges)} edges")
    print(f"{rooms} numbered rooms | {len(vertical_report)} vertical links "
          f"| {outdoor_stats.get('edges', 0)} outdoor edges")
    print(f"connected components: {comps}")
    total_edges_checked = sum(1 for _, _, d in graph.edges(data=True)
                              if d.get("kind") in ("room_to_door", "door_to_corridor",
                                                   "room_to_room", "corridor_access",
                                                   "inferred_access"))
    print(f"edges through walls: {len(wall_crossings)} "
          f"(of {total_edges_checked} indoor connector edges)")
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB, "
          f"{os.path.getsize(path+'.gz')/1e6:.1f} MB gzipped)")


if __name__ == "__main__":
    main()
