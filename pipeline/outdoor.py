"""
The outdoor half of the campus graph: real pedestrian paths between buildings.

Why this file exists
--------------------
Getting from a room in one building to a room in another means walking outside,
and that leg has to be as honest as the indoor ones. The previous attempt at
this project joined buildings with straight lines between their centres and
multiplied the distance by 1.25 to account for the fact that you cannot
actually walk through whatever is in between. That is a guess wearing a
decimal point: it routes you through buildings, across the middle of a field,
and over University Drive.

This module uses the paths people actually walk, from OpenStreetMap:
`highway=footway | path | steps | pedestrian | living_street`, plus the
sidewalk-tagged parts of the road network, for the campus bounding box.

How it connects to the indoor graph
-----------------------------------
Each building contributes one entrance node (chosen in build_campus.py as the
ground-floor corridor point nearest the outer wall). That entrance is joined to
the nearest point on the OSM path network. If the nearest path is further away
than MAX_ENTRANCE_SNAP_M the building is left unconnected rather than attached
by a long invented line, and it is reported as such.

`steps` ways are tagged so accessible routing can avoid them, the same way it
avoids stairs indoors.

What breaks if the assumptions don't hold
-----------------------------------------
OSM coverage of campus paths is good but not perfect, and a missing footway
means two buildings may route via a longer way round. That is a wrong-but-
walkable answer rather than a wrong-and-impossible one, which is the right
failure mode. Buildings that snap to nothing are listed in the build log.

The Overpass response is cached in data/osm-paths.json so reruns don't hammer
a public API; delete that file to refetch.
"""
import json
import math
import os
import urllib.parse
import urllib.request

import networkx as nx
from shapely.geometry import LineString, Point

from georeference import to_wgs84
import pyproj

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
)

# Campus bounding box (lat/lon), wide enough to take in main campus, Foothills
# and the Spy Hill / veterinary site.
CAMPUS_BBOX = (50.98, -114.25, 51.19, -114.05)

# Pedestrian ways only. An earlier version also pulled service roads,
# residential streets, unclassified roads and tracks, which dragged in most of
# Calgary's road network: 303,817 nodes and a 74 MB graph, nearly all of it
# streets nobody walks between lectures on.
WALKABLE = ("footway", "path", "steps", "pedestrian", "living_street")

MAX_ENTRANCE_SNAP_M = 120.0

# Keep OSM paths within this distance of a building we have plans for.
NEAR_BUILDING_M = 450.0

_to_3780 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3780", always_xy=True)


def _to_metres(lon, lat):
    return _to_3780.transform(lon, lat)


def fetch_paths(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            return json.load(fh)
    south, west, north, east = CAMPUS_BBOX
    filt = "|".join(WALKABLE)
    # One filtered clause only. An earlier version added a bare
    # way["footway"] clause as a catch-all, which has no highway filter to
    # narrow it and made Overpass time out with a 504 over a bbox this size.
    query = (f'[out:json][timeout:180];'
             f'way["highway"~"^({filt})$"]({south},{west},{north},{east});'
             f'out geom;')
    data = urllib.parse.urlencode({"data": query}).encode()

    last_error = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            req = urllib.request.Request(
                endpoint, data=data,
                headers={"User-Agent": "ucalgary-wayfinder/1.0 (student project)"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            with open(cache_path, "w") as fh:
                json.dump(payload, fh)
            return payload
        except Exception as e:
            last_error = e
            print(f"  [warn] Overpass {endpoint} failed ({type(e).__name__}); trying next")
    raise RuntimeError(f"all Overpass endpoints failed: {last_error}")


def build_path_graph(payload):
    """
    OSM ways -> a metric graph in EPSG:3780.

    Ways share nodes at junctions; keying graph nodes on rounded coordinates
    joins them up without needing the OSM node ids.
    """
    graph = nx.Graph()
    for element in payload.get("elements", []):
        if element.get("type") != "way":
            continue
        geometry = element.get("geometry") or []
        if len(geometry) < 2:
            continue
        tags = element.get("tags", {})
        is_steps = tags.get("highway") == "steps"
        points = [_to_metres(p["lon"], p["lat"]) for p in geometry]
        for a, b in zip(points, points[1:]):
            ka = (round(a[0], 1), round(a[1], 1))
            kb = (round(b[0], 1), round(b[1], 1))
            if ka == kb:
                continue
            graph.add_node(ka, pos=ka)
            graph.add_node(kb, pos=kb)
            graph.add_edge(ka, kb, weight=math.dist(ka, kb),
                           kind="steps" if is_steps else "outdoor")
    return _collapse_chains(graph)


def _collapse_chains(graph):
    """
    Squash runs of degree-2 nodes into single edges.

    OSM gives a node at every vertex of every way, so a gently curving path
    becomes dozens of nodes that carry no routing choice. Keeping only
    junctions and dead ends leaves the same network with a fraction of the
    nodes, and the edge weights still add up to the true walking distance
    because the collapsed lengths are summed.
    """
    import networkx as nx
    changed = True
    while changed:
        changed = False
        for node in [n for n in list(graph.nodes) if graph.degree(n) == 2]:
            if not graph.has_node(node) or graph.degree(node) != 2:
                continue
            a, b = list(graph.neighbors(node))
            if a == b or graph.has_edge(a, b):
                continue
            w = graph.edges[a, node]["weight"] + graph.edges[node, b]["weight"]
            kinds = {graph.edges[a, node]["kind"], graph.edges[node, b]["kind"]}
            graph.remove_node(node)
            graph.add_edge(a, b, weight=w,
                           kind="steps" if "steps" in kinds else "outdoor")
            changed = True
    return graph


def connect(campus, buildings, entrances, cache_dir):
    """
    Merge the OSM walking network into `campus` and hook each building on.
    Returns a stats dict for the build report.
    """
    cache = os.path.join(cache_dir, "osm-paths.json")
    try:
        payload = fetch_paths(cache)
    except Exception as e:
        print(f"[warn] could not fetch OSM paths ({e}); buildings will not be "
              "connected to each other")
        return {"edges": 0, "error": str(e)[:200]}

    paths = build_path_graph(payload)
    if paths.number_of_nodes() == 0:
        return {"edges": 0, "error": "no walkable ways returned"}

    # The bounding box has to be wide enough to take in Foothills and Spy Hill
    # as well as main campus, which means it also takes in a large slice of
    # north-west Calgary - tens of thousands of nodes of suburban footpath
    # that no campus route will ever touch. Keep only what is near a building
    # we actually have floor plans for.
    keep_near = []
    for info in buildings.values():
        keep_near.append(_to_metres(info["lon"], info["lat"]))
    if keep_near:
        relevant = [key for key in paths.nodes
                    if any(math.dist(key, c) <= NEAR_BUILDING_M for c in keep_near)]
        paths = paths.subgraph(relevant).copy()

    # then the largest connected chunk - isolated fragments would only add noise
    if paths.number_of_nodes() == 0:
        return {"edges": 0, "error": "no walkable ways near any building"}
    biggest = max(nx.connected_components(paths), key=len)
    paths = paths.subgraph(biggest).copy()

    index = {}
    for key, data in paths.nodes(data=True):
        node_id = f"osm|{key[0]:.1f}|{key[1]:.1f}"
        index[key] = node_id
        campus.add_node(node_id, pos=key, kind="outdoor",
                        building=None, floor=None)
    for a, b, data in paths.edges(data=True):
        campus.add_edge(index[a], index[b], weight=data["weight"],
                        kind=data["kind"], geometry=None)

    node_keys = list(paths.nodes)
    connected, unconnected = 0, []
    for code, entrance in entrances.items():
        if not campus.has_node(entrance):
            continue
        ex, ey = campus.nodes[entrance]["pos"]
        best, best_d = None, None
        for key in node_keys:
            d = math.dist((ex, ey), key)
            if best_d is None or d < best_d:
                best, best_d = key, d
        if best is None or best_d > MAX_ENTRANCE_SNAP_M:
            unconnected.append({"building": code,
                                "nearest_path_m": round(best_d, 1) if best_d else None})
            continue
        campus.add_edge(entrance, index[best], weight=max(best_d, 1.0),
                        kind="entrance_link",
                        geometry=LineString([(ex, ey), best]))
        connected += 1

    stats = {
        "osm_nodes": paths.number_of_nodes(),
        "osm_ways_edges": paths.number_of_edges(),
        "buildings_connected": connected,
        "buildings_unconnected": unconnected,
        "edges": paths.number_of_edges() + connected,
    }
    print(f"  outdoor: {paths.number_of_nodes()} path nodes, "
          f"{paths.number_of_edges()} path edges, {connected} buildings linked"
          + (f", {len(unconnected)} unlinked" if unconnected else ""))
    return stats
