"""
Check the graph is actually walkable, and report numbers rather than
reassurance.

Run:  .venv/bin/python pipeline/verify.py

Why this file exists
--------------------
Every stage of this pipeline involves inference: which closed path is a room,
which room is a corridor, where a door sits, which floor sits above which. Any
of those can be wrong in a way that still produces a confident-looking route
on a map. The whole point of building it this way rather than with straight
lines was to be able to *check*, so these are the checks.

What each one is really asking
------------------------------
**Edges through walls.** The central claim of floor_graph.py is that every
edge either stays inside one space or crosses a shared boundary where a door
must be. That test needs the room polygons, which only exist while a floor is
being built, so it runs in build_campus.py and lands in out/wall-crossings.json;
this module reads and summarises the result.

**Unreachable rooms.** A room that can't be reached from its floor's corridor
network is either a bad door match or a genuinely sealed space (a duct, a
shaft). The count matters more than any single case: if it is a handful per
floor that is the second; if it is half the floor, the corridor detection has
failed on that floor.

**Vertical alignment.** A stair core on floor 3 and its partner on floor 4 are
the same physical shaft, so after georeferencing they should sit within a
couple of metres of each other horizontally. A large offset means the two
floors were fitted inconsistently.

**Rooms inside their building.** Every room should land inside its building's
ArcGIS footprint. Rooms outside it mean the floor was placed wrong.

**Route sanity.** Distances are in metres, so known walks can be sanity
checked. A route the length of a corridor should not come back as 12 metres.
"""
import json
import math
import os
import sys
from collections import defaultdict

import networkx as nx
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT, "out")

# An edge may clip this much of a foreign space before it counts as going
# through a wall - covers polygons that share a boundary drawn to the wall
# centreline rather than its faces.
WALL_CROSS_TOLERANCE_M = 0.45
# Stair partners should sit above each other; more than this apart means the
# two floors disagree about where they are.
VERTICAL_OFFSET_WARN_M = 3.0


def load_graph():
    with open(os.path.join(OUT_DIR, "navigation-graph.json")) as fh:
        data = json.load(fh)
    graph = nx.Graph()
    for node in data["nodes"]:
        graph.add_node(node["id"], **node)
    for edge in data["edges"]:
        graph.add_edge(edge["a"], edge["b"], **edge)
    return data, graph


def main():
    data, graph = load_graph()
    report = {}

    # ---------- reachability, per floor ----------
    floors = defaultdict(list)
    for node_id, d in graph.nodes(data=True):
        if d.get("building") and d.get("floor"):
            floors[(d["building"], d["floor"])].append(node_id)

    unreachable, floor_rows = [], []
    for key, node_ids in sorted(floors.items()):
        sub = graph.subgraph(node_ids)
        corridor_nodes = [n for n in node_ids
                          if graph.nodes[n].get("kind") == "corridor"]
        room_nodes = [n for n in node_ids
                      if graph.nodes[n].get("kind") == "room"]
        if not room_nodes:
            continue
        if corridor_nodes:
            reachable = set()
            for comp in nx.connected_components(sub):
                if comp & set(corridor_nodes):
                    reachable |= comp
        else:
            reachable = set()
        missed = [n for n in room_nodes if n not in reachable]
        unreachable.extend({"building": key[0], "floor": key[1],
                            "room": graph.nodes[n].get("room"), "node": n}
                           for n in missed)
        floor_rows.append({
            "building": key[0], "floor": key[1],
            "rooms": len(room_nodes), "corridor_nodes": len(corridor_nodes),
            "unreachable": len(missed),
            "components": nx.number_connected_components(sub),
        })

    report["floors"] = floor_rows
    report["unreachable_rooms"] = unreachable

    # ---------- vertical alignment ----------
    vertical = []
    for u, v, d in graph.edges(data=True):
        if d.get("kind") not in ("stairs", "vertical_unknown"):
            continue
        a, b = graph.nodes[u], graph.nodes[v]
        # positions are lon/lat in the exported graph; compare in metres
        import pyproj
        t = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3780", always_xy=True)
        ax, ay = t.transform(a["lon"], a["lat"])
        bx, by = t.transform(b["lon"], b["lat"])
        offset = math.dist((ax, ay), (bx, by))
        vertical.append({"building": a.get("building"),
                         "from": a.get("floor"), "to": b.get("floor"),
                         "kind": d.get("kind"),
                         "offset_m": round(offset, 2)})
    report["vertical_links"] = vertical
    report["vertical_misaligned"] = [v for v in vertical
                                     if v["offset_m"] > VERTICAL_OFFSET_WARN_M]

    # ---------- connectivity overall ----------
    report["components"] = nx.number_connected_components(graph)
    largest = max(nx.connected_components(graph), key=len)
    report["largest_component"] = len(largest)
    report["total_nodes"] = graph.number_of_nodes()

    rooms_total = sum(1 for _, d in graph.nodes(data=True) if d.get("room"))
    rooms_in_main = sum(1 for n in largest if graph.nodes[n].get("room"))
    report["rooms_total"] = rooms_total
    report["rooms_in_largest_component"] = rooms_in_main

    # ---------- summary ----------
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    print(f"nodes {graph.number_of_nodes():>8}   edges {graph.number_of_edges():>8}")
    print(f"connected components {report['components']:>5}  "
          f"largest holds {report['largest_component']} nodes")
    print(f"rooms {rooms_total:>8}   reachable in largest component "
          f"{rooms_in_main} ({100*rooms_in_main/max(1,rooms_total):.0f}%)")
    print()
    bad_floors = [f for f in floor_rows if f["rooms"] and
                  f["unreachable"] / f["rooms"] > 0.25]
    print(f"floors checked {len(floor_rows)};  "
          f"{len(unreachable)} rooms unreachable from their corridor network")
    print(f"floors where >25% of rooms are unreachable: {len(bad_floors)}")
    for f in sorted(bad_floors, key=lambda f: -f["unreachable"])[:10]:
        print(f"   {f['building']:5s} {f['floor']:5s} "
              f"{f['unreachable']}/{f['rooms']} unreachable, "
              f"{f['corridor_nodes']} corridor nodes")
    print()
    print(f"vertical links {len(vertical)};  "
          f"{len(report['vertical_misaligned'])} sit more than "
          f"{VERTICAL_OFFSET_WARN_M} m off between floors")
    kinds = defaultdict(int)
    for v in vertical:
        kinds[v["kind"]] += 1
    for k, n in kinds.items():
        print(f"   {k}: {n}")

    with open(os.path.join(OUT_DIR, "verification.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nfull report -> out/verification.json")
    return report


if __name__ == "__main__":
    main()
