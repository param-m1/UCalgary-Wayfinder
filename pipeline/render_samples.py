"""
Draw sample floors with their rooms, corridors, doors and routing graph on top.

Run:  .venv/bin/python pipeline/render_samples.py [--count 12]

Why this file exists
--------------------
Assertions catch the errors you thought to test for. Pictures catch the rest.
Nearly every real mistake in this project was caught by looking at one:

  * the georeferencing scored a flawless 1.000 on every Art Building floor
    while placing each of them at about 70% of its true size - obvious the
    moment eight floors were drawn inside their footprint
  * the corridor classifier turned 73% of Biological Sciences floor 1 blue,
    which no amount of per-floor statistics had made me suspicious of

So this writes a set of floor renders to renders/ for flipping through. Each
one shows corridors in blue, rooms in grey, the corridor centreline, the
room-to-door and door-to-corridor connections, and every inferred door
position, with the counts in the title.
"""
import argparse
import glob
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Polygon
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import floor_graph as fg
from georeference import (BUILDING_CODE_ALIASES, apply_transform, fetch_footprints,
                          floor_outline, page_polygons, rings_to_polygon,
                          solve_building)

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOOR_DIR = os.path.join(PROJECT, "data", "floors")
RENDER_DIR = os.path.join(PROJECT, "renders", "floors")

EDGE_COLOURS = {
    "corridor": "#1a6fd4",
    "room_to_door": "#d62728",
    "door_to_corridor": "#2ca02c",
    "room_to_room": "#9467bd",
    "corridor_access": "#8c564b",
    "inferred_access": "#ff7f0e",
}


def render_floor(code, floor_id, footprint, params, rooms, out_path):
    graph, stats, spaces = fg.build(code, floor_id, rooms)
    corridors, others = fg.classify_spaces(
        [r for r in rooms if r["poly"].area > 0.5])
    circulation = unary_union([c["poly"] for c in corridors]) if corridors else None
    violations = fg.edges_through_walls(graph, spaces, circulation)

    fig, ax = plt.subplots(figsize=(16, 12))
    for geom in ([footprint] if footprint.geom_type == "Polygon" else list(footprint.geoms)):
        ax.plot(*geom.exterior.xy, color="#e11", lw=1.6, alpha=0.6, zorder=1)
    for rec in others:
        ax.fill(*rec["poly"].exterior.xy, color="#fafafa", ec="#c9ced6", lw=0.4, zorder=2)
    for rec in corridors:
        ax.fill(*rec["poly"].exterior.xy, color="#d6e9ff", ec="#4a90d9", lw=0.6, zorder=3)
    for rec in others:
        if rec.get("number"):
            pt = rec["poly"].representative_point()
            ax.text(pt.x, pt.y, rec["number"], fontsize=4.2, ha="center", va="center",
                    color="#444", zorder=8)

    for u, v, data in graph.edges(data=True):
        geom = data.get("geometry")
        if geom is None:
            continue
        ax.plot(*geom.xy, color=EDGE_COLOURS.get(data.get("kind"), "#333"),
                lw=1.3 if data.get("kind") == "corridor" else 0.9, zorder=6)

    doors = [d["pos"] for _, d in graph.nodes(data=True) if d.get("kind") == "door"]
    if doors:
        ax.scatter([p[0] for p in doors], [p[1] for p in doors], s=9, c="#d62728",
                   zorder=9, label=f"doors ({len(doors)})")
    inferred = [d["pos"] for _, d in graph.nodes(data=True) if d.get("door_inferred")]
    if inferred:
        ax.scatter([p[0] for p in inferred], [p[1] for p in inferred], s=26, c="#ff7f0e",
                   marker="x", zorder=10, label=f"inferred ({len(inferred)})")

    components = nx.number_connected_components(graph) if graph.number_of_nodes() else 0
    corridor_area = sum(c["poly"].area for c in corridors)
    total_area = corridor_area + sum(o["poly"].area for o in others)
    ax.set_title(
        f"{code} {floor_id}   "
        f"{len(others)} rooms, {len(corridors)} corridors "
        f"({100*corridor_area/max(1,total_area):.0f}% of area)   "
        f"{stats['doors']} doors, {stats['room_to_room']} room-to-room, "
        f"{stats['inferred_doors']} inferred   "
        f"{components} graph components   "
        f"{len(violations)} edges through walls   "
        f"fit IoU {params['score']:.2f}",
        fontsize=11)
    if doors or inferred:
        ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=95, bbox_inches="tight")
    plt.close(fig)
    return stats, len(violations), components


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--only", default=None, help="one building code")
    args = ap.parse_args()

    os.makedirs(RENDER_DIR, exist_ok=True)
    footprints = fetch_footprints(os.path.join(PROJECT, "data", "footprints.json"))

    codes = sorted(os.path.basename(p) for p in glob.glob(os.path.join(FLOOR_DIR, "*")))
    if args.only:
        codes = [args.only]
    else:
        random.seed(args.seed)
        codes = random.sample(codes, min(args.count, len(codes)))

    for code in codes:
        key = BUILDING_CODE_ALIASES.get(code, code)
        if key not in footprints:
            continue
        footprint = rings_to_polygon(footprints[key]["rings"])
        floors, data = [], {}
        for path in sorted(glob.glob(os.path.join(FLOOR_DIR, code, "*.json"))):
            with open(path) as fh:
                d = json.load(fh)
            outline = floor_outline(d)
            if outline is not None and outline.area > 0:
                floors.append((d["floor_id"], outline))
                data[d["floor_id"]] = d
        if not floors:
            continue
        rotation, ref, fits = solve_building(
            floors, footprint,
            floor_polys={fid: page_polygons(data[fid]) for fid in data})

        # the floor with the most rooms is the most informative to look at
        floor_id = max(data, key=lambda f: len(data[f].get("rooms", [])))
        if floor_id not in fits:
            continue
        params = fits[floor_id]
        rooms = []
        for room in data[floor_id].get("rooms", []):
            ring = room.get("ring")
            if not ring or len(ring) < 4:
                continue
            try:
                g = apply_transform(Polygon(ring), params)
                if g.is_valid and not g.is_empty and g.area > 0.5:
                    rooms.append({"number": room.get("number"), "poly": g})
            except Exception:
                continue
        if not rooms:
            continue
        out_path = os.path.join(RENDER_DIR, f"{code}-{floor_id}.png")
        try:
            stats, bad, comps = render_floor(code, floor_id, footprint, params,
                                             rooms, out_path)
        except Exception as e:
            print(f"  {code} {floor_id}: render failed ({type(e).__name__}: {e})")
            continue
        print(f"  {code:5s} {floor_id:5s} rooms={stats['rooms']:4d} doors={stats['doors']:4d} "
              f"r2r={stats['room_to_room']:3d} inferred={stats['inferred_doors']:3d} "
              f"comps={comps:3d} through_walls={bad:3d} -> {os.path.basename(out_path)}")

    print(f"\nrenders in {RENDER_DIR}")


if __name__ == "__main__":
    main()
