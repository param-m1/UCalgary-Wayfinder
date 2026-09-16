"""
One floor's room polygons -> a routable graph you can actually walk.

Why this file exists
--------------------
This is the part that decides whether a route is true or merely plausible. The
previous attempt at this project connected each room to the corridor network
with a straight line from the middle of the room to the nearest corridor
point. That line goes through whatever happens to be in the way - a wall, a
neighbouring office, a stairwell - so the resulting route looks reasonable on
a map and is wrong on the ground.

Having real room polygons makes it possible to do this properly, because two
adjoining spaces share a boundary, and **a room's door has to be on the
boundary it shares with a corridor**. So instead of guessing a line, we:

  1. work out which spaces are circulation and which are rooms,
  2. build a centreline down the middle of the circulation space,
  3. attach each room to that centreline *at the boundary it shares with the
     corridor* - the doorway.

Every edge then lies inside a space or crosses a room/corridor interface where
a door must be. No edge crosses a wall, by construction rather than by luck.

Finding the corridors
---------------------
This is the crux, and it is the one piece of knowledge worth carrying over
wholesale from the previous pipeline, because it was expensive to learn:

**corridors in this dataset are mapped as rooms in their own right**, with
their own room numbers (a corridor is typically numbered like `199Z`). The
intuitive approach - take the floor outline and subtract the rooms, whatever
is left is corridor - therefore leaves almost nothing. Measured on
Administration floor 1 it left 392 m2 of a 4,290 m2 floor and produced a
16-metre "corridor network" for an entire building.

So circulation is identified from three independent signals, combined so that
no single convention has to hold everywhere:

  * **the `Z` room-number suffix**, which this dataset uses for circulation
  * **adjacency degree** - used only to confirm an already-elongated shape;
    on its own it is not diagnostic (see the note by the constants)
  * **circularity** - corridors are long, thin and branched; rooms are blobby.
    Circularity is 4*pi*area / perimeter^2, which is 1.0 for a circle and
    tends towards 0 for a long thin shape.

Doors
-----
There is no door layer to read, so a door is inferred as the *shared boundary
between a room and a corridor*. Where a room and a corridor touch along a
stretch of wall, the door is placed at the midpoint of that shared stretch.

This is an inference, and it is worth being precise about what it does and
doesn't get right. It reliably puts the connection **on the wall between the
two spaces**, so a route never cuts through a third space to reach a corridor.
It does not know *where along* that wall the real door sits, so for a room
with a long corridor frontage the door may be placed a few metres from the
true opening.

Plenty of rooms have no corridor frontage at all - roughly a quarter of them -
and that is normal rather than broken: a room in a suite, or a lab behind an
ante-room, opens into another room and reaches the corridor through it. The
first version of this fell back to a straight line to the nearest corridor
centreline point for those, which is precisely the through-the-wall behaviour
this module exists to prevent. They now connect through the shared boundary
with whichever neighbouring room they abut most, so that edge is a real
doorway on a real shared wall too, and the route reaches the corridor via the
neighbour. Only a room that touches nothing at all falls back to a straight
line, and it is tagged `door_inferred` so it can be counted and reported.

What breaks if the assumptions don't hold
-----------------------------------------
A floor whose corridors are not numbered with `Z` and are not notably
elongated - a single open-plan space, a small mechanical level - may end up
with no circulation identified. Those floors produce a graph of rooms with no
corridor spine, which shows up in verification as rooms unreachable from
circulation rather than as a confidently wrong route.
"""
import math
import re

import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

# --- circulation classification -------------------------------------------
CORRIDOR_SUFFIX_RE = re.compile(r"^\d+\s*Z$", re.IGNORECASE)
ADJACENCY_TOLERANCE_M = 0.30

# Adjacency degree on its own is NOT a usable corridor signal here, and this
# is a correction to the inherited heuristic rather than a copy of it.
#
# The previous pipeline treated "touches 6+ other spaces" as enough to call
# something circulation. That was calibrated against geometry where rooms did
# not share exact boundaries. With true room polygons every room shares walls
# with its neighbours, so ordinary rooms routinely score 7-13. On Biological
# Sciences floor 1 the old rule called room 186 a corridor - a 321 m2 space
# with circularity 0.51, which is to say a large blobby lab - along with ten
# others like it, and ended up classifying 73% of the floor as circulation.
#
# Measured over four floors, corridor share of floor area:
#     rule                              BI-01   AD-01   AU-03   BI-04
#     Z | deg>=6 | deg>=4 & circ<.30    73.2%   52.6%   39.3%   60.3%
#     Z suffix only                     21.6%   34.2%   21.3%   18.7%
#     Z | circ<.30 & deg>=3             22.2%   35.9%   21.3%   19.4%
#
# 15-35% is the plausible range for circulation in a building, so degree is
# kept only as a secondary condition on shapes that are already elongated.
ELONGATED_CIRCULARITY = 0.30
MIN_DEGREE_FOR_ELONGATED = 3

# --- door inference --------------------------------------------------------
# Two spaces are "touching" if their boundaries come within this distance.
# Room polygons are drawn to the centre of the wall, so adjoining spaces
# usually share an edge exactly; the tolerance covers small CAD gaps.
SHARED_EDGE_TOLERANCE_M = 0.35
# A shared stretch shorter than this is a corner touch, not a doorway.
MIN_DOOR_FRONTAGE_M = 0.60
# A room with no corridor frontage gets connected to the nearest corridor
# point, but not across an absurd distance.
MAX_FALLBACK_CONNECT_M = 25.0

# --- corridor centreline ---------------------------------------------------
CENTRELINE_RESOLUTION_M = 0.35
# Prune skeleton whiskers shorter than this; they are rasterisation noise
# poking into doorways and wall nooks, not real corridor branches.
SPUR_PRUNE_M = 1.5


def circularity(poly):
    if poly.length <= 0:
        return 0.0
    return 4 * math.pi * poly.area / (poly.length ** 2)


def classify_spaces(rooms):
    """
    rooms: list of dicts {"number": str|None, "poly": shapely Polygon}
    Returns (corridors, others) as two lists of the same dicts, each annotated
    with the signals that decided it.
    """
    buffered = [r["poly"].buffer(ADJACENCY_TOLERANCE_M) for r in rooms]
    degrees = [0] * len(rooms)
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if buffered[i].intersects(rooms[j]["poly"]):
                degrees[i] += 1
                degrees[j] += 1

    corridors, others = [], []
    for rec, deg in zip(rooms, degrees):
        number = (rec.get("number") or "").upper().replace(" ", "")
        circ = circularity(rec["poly"])
        by_suffix = bool(CORRIDOR_SUFFIX_RE.match(number))
        is_corridor = (
            by_suffix
            or (circ < ELONGATED_CIRCULARITY and deg >= MIN_DEGREE_FOR_ELONGATED)
        )
        rec = dict(rec, degree=deg, circularity=round(circ, 3),
                   corridor_by_suffix=by_suffix)
        (corridors if is_corridor else others).append(rec)
    return corridors, others


# --- corridor centreline ---------------------------------------------------

class _Raster:
    def __init__(self, bounds, resolution, pad=4):
        self.minx, self.miny, self.maxx, self.maxy = bounds
        self.res = resolution
        self.pad = pad
        self.width = int(math.ceil((self.maxx - self.minx) / resolution)) + 2 * pad
        self.height = int(math.ceil((self.maxy - self.miny) / resolution)) + 2 * pad

    def to_px(self, x, y):
        return ((x - self.minx) / self.res + self.pad,
                (self.maxy - y) / self.res + self.pad)

    def to_world(self, col, row):
        return (self.minx + (col - self.pad) * self.res,
                self.maxy - (row - self.pad) * self.res)


def _polygons(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def centreline(circulation, resolution=CENTRELINE_RESOLUTION_M):
    """
    A walkable centreline down the middle of the circulation space.

    Done by rasterising the circulation polygon and thinning it to a
    one-pixel-wide skeleton, which is the standard way to get a medial axis
    for a polygon of this shape complexity. The alternative - a Voronoi
    diagram of the boundary vertices - was what the previous pipeline used,
    and it emitted one tiny disconnected segment per Voronoi edge: 16,833
    segments for a single floor, which is line soup rather than a graph.
    """
    from PIL import Image, ImageDraw
    from skimage.morphology import skeletonize
    import sknw

    graph = nx.Graph()
    polys = _polygons(circulation)
    if not polys:
        return graph

    raster = _Raster(circulation.bounds, resolution)
    if raster.width * raster.height > 30_000_000:
        return graph

    # 8-bit mode, not PIL's 1-bit "1" mode: a bit-packed array crashes
    # skimage's skeletonize with a bus error rather than an exception.
    img = Image.new("L", (raster.width, raster.height), 0)
    draw = ImageDraw.Draw(img)
    for poly in polys:
        ext = [raster.to_px(x, y) for x, y in poly.exterior.coords]
        if len(ext) >= 3:
            draw.polygon(ext, fill=255)
        for hole in poly.interiors:
            h = [raster.to_px(x, y) for x, y in hole.coords]
            if len(h) >= 3:
                draw.polygon(h, fill=0)

    mask = np.ascontiguousarray(np.asarray(img, dtype=np.uint8)) > 0
    if not mask.any():
        return graph
    skel = skeletonize(mask)
    if not skel.any():
        return graph

    sk = sknw.build_sknw(skel, multi=False)
    for node, data in sk.nodes(data=True):
        row, col = data["o"]
        graph.add_node(int(node), pos=raster.to_world(col, row))
    for u, v, data in sk.edges(data=True):
        if u == v:
            continue
        coords = [raster.to_world(c, r) for r, c in data["pts"]]
        coords = ([graph.nodes[int(u)]["pos"]] + coords + [graph.nodes[int(v)]["pos"]])
        deduped = [coords[0]]
        for c in coords[1:]:
            if c != deduped[-1]:
                deduped.append(c)
        if len(deduped) < 2:
            continue
        line = LineString(deduped)
        graph.add_edge(int(u), int(v), geometry=line, weight=line.length)

    _prune_spurs(graph)
    return graph


def _prune_spurs(graph, min_length=SPUR_PRUNE_M):
    changed = True
    while changed:
        changed = False
        for node in [n for n in list(graph.nodes) if graph.degree(n) == 1]:
            # degrees shift as nodes are removed; re-check before trusting
            if not graph.has_node(node) or graph.degree(node) != 1:
                continue
            neighbour = next(iter(graph.neighbors(node)))
            if graph.edges[node, neighbour]["weight"] < min_length:
                graph.remove_node(node)
                changed = True
    graph.remove_nodes_from([n for n in list(graph.nodes) if graph.degree(n) == 0])


# --- doors -----------------------------------------------------------------

def find_door(room_poly, corridor_poly):
    """
    Where a room meets a corridor, return (door_point, frontage_length).

    The shared boundary is found by intersecting the room's boundary with a
    slightly thickened corridor: where the two spaces abut, the room's wall
    runs along the corridor. The door goes at the midpoint of the longest
    shared stretch. Returns None if they only touch at a corner.
    """
    try:
        shared = room_poly.exterior.intersection(
            corridor_poly.buffer(SHARED_EDGE_TOLERANCE_M))
    except Exception:
        return None
    if shared.is_empty:
        return None

    pieces = []
    if shared.geom_type == "LineString":
        pieces = [shared]
    elif shared.geom_type in ("MultiLineString", "GeometryCollection"):
        pieces = [g for g in shared.geoms if g.geom_type == "LineString"]
    if not pieces:
        return None

    longest = max(pieces, key=lambda g: g.length)
    if longest.length < MIN_DOOR_FRONTAGE_M:
        return None
    return longest.interpolate(longest.length / 2), longest.length


def _substring(line, start, end):
    """The piece of `line` between two distances along it."""
    start = max(0.0, min(start, line.length))
    end = max(0.0, min(end, line.length))
    if end - start < 1e-9:
        p = line.interpolate(start)
        return LineString([(p.x, p.y), (p.x + 1e-6, p.y)])
    coords = [line.interpolate(start)]
    for pt in line.coords:
        d = line.project(Point(pt))
        if start < d < end:
            coords.append(Point(pt))
    coords.append(line.interpolate(end))
    pts = [(p.x, p.y) for p in coords]
    deduped = [pts[0]]
    for q in pts[1:]:
        if q != deduped[-1]:
            deduped.append(q)
    if len(deduped) < 2:
        deduped = [pts[0], (pts[0][0] + 1e-6, pts[0][1])]
    return LineString(deduped)


def build(building_code, floor_id, rooms):
    """
    rooms: list of {"number": str|None, "poly": Polygon} in real-world metres.

    Returns (graph, stats, spaces) where spaces is [(node_id, polygon)] so
    callers can check edges against the actual room geometry. Node ids are
    handed back explicitly rather than reconstructed by position, because
    corridors and rooms are added in a different order to the input list and
    pairing them up by index silently mismatches them.

    Graph nodes carry:
        pos    (x, y) in EPSG:3780 metres
        kind   "room" | "corridor" | "door"
        number room number, for room nodes
    Edges carry weight (metres), geometry, and a kind.
    """
    prefix = f"{building_code}|{floor_id}|"
    graph = nx.Graph()

    usable = [r for r in rooms
              if r["poly"] is not None and not r["poly"].is_empty and r["poly"].area > 0.5]
    if not usable:
        return graph, {"rooms": 0, "corridors": 0, "doors": 0,
                       "room_to_room": 0, "inferred_doors": 0}, []

    corridors, others = classify_spaces(usable)

    # ---- corridor spine
    circulation = unary_union([c["poly"] for c in corridors]) if corridors else None
    spine = centreline(circulation) if circulation is not None else nx.Graph()

    for node, data in spine.nodes(data=True):
        graph.add_node(f"{prefix}c{node}", pos=data["pos"], kind="corridor")
    for u, v, data in spine.edges(data=True):
        graph.add_edge(f"{prefix}c{u}", f"{prefix}c{v}", weight=data["weight"],
                       geometry=data["geometry"], kind="corridor")

    # Attach to the nearest POINT on the corridor centreline, not the nearest
    # centreline *node*.
    #
    # Connecting to the nearest node looks equivalent and is not: skeleton
    # nodes are junctions and dead ends, so the nearest one to a given door can
    # be tens of metres away down the corridor, and the straight line to it
    # cuts through every room in between. Measured across 30 buildings, that
    # version put 11,385 of 17,713 indoor edges through a wall. Splitting the
    # nearest centreline edge at the closest point instead makes the connection
    # a short hop perpendicular into the corridor, which is what a doorway
    # actually is.
    attach_counter = [0]

    def attach(point):
        """Returns (node_id, distance) for the closest spot on the spine."""
        best = None
        for u, v, data in list(graph.edges(data=True)):
            if data.get("kind") != "corridor":
                continue
            line = data.get("geometry")
            if line is None or line.length <= 0:
                continue
            d = line.distance(point)
            if best is None or d < best[0]:
                best = (d, u, v, line)
        if best is None:
            return None, None
        dist, u, v, line = best

        along = line.project(point)
        # reuse an existing endpoint when the projection lands on one, rather
        # than creating a zero-length stub
        if along <= 0.05:
            return u, dist
        if along >= line.length - 0.05:
            return v, dist

        snapped = line.interpolate(along)
        node_id = f"{prefix}s{attach_counter[0]}"
        attach_counter[0] += 1
        graph.add_node(node_id, pos=(snapped.x, snapped.y), kind="corridor")

        first = _substring(line, 0.0, along)
        second = _substring(line, along, line.length)
        graph.remove_edge(u, v)
        graph.add_edge(u, node_id, weight=max(first.length, 0.01),
                       geometry=first, kind="corridor")
        graph.add_edge(node_id, v, weight=max(second.length, 0.01),
                       geometry=second, kind="corridor")
        return node_id, dist

    def nearest_spine(point):
        return attach(point)

    stats = {"rooms": 0, "corridors": len(corridors), "doors": 0,
             "room_to_room": 0, "inferred_doors": 0}
    spaces = []

    # ---- corridor polygons themselves get a node so corridor-to-corridor
    #      movement still works on floors where the spine is thin
    for i, corridor in enumerate(corridors):
        pt = corridor["poly"].representative_point()
        node_id = f"{prefix}k{i}"
        graph.add_node(node_id, pos=(pt.x, pt.y), kind="room",
                       number=corridor.get("number"), is_corridor=True)
        spaces.append((node_id, corridor["poly"]))
        stats["rooms"] += 1
        target, dist = nearest_spine(pt)
        if target is not None and dist is not None:
            graph.add_edge(node_id, target, weight=max(dist, 0.1), kind="corridor_access",
                           geometry=LineString([(pt.x, pt.y), graph.nodes[target]["pos"]]))

    # ---- rooms attach through their doorway
    for i, room in enumerate(others):
        pt = room["poly"].representative_point()
        node_id = f"{prefix}r{i}"
        graph.add_node(node_id, pos=(pt.x, pt.y), kind="room",
                       number=room.get("number"), is_corridor=False)
        spaces.append((node_id, room["poly"]))
        stats["rooms"] += 1

        best_door = None
        for corridor in corridors:
            found = find_door(room["poly"], corridor["poly"])
            if found is None:
                continue
            door_pt, frontage = found
            if best_door is None or frontage > best_door[1]:
                best_door = (door_pt, frontage)

        if best_door is not None:
            door_pt, frontage = best_door
            door_id = f"{prefix}d{i}"
            graph.add_node(door_id, pos=(door_pt.x, door_pt.y), kind="door",
                           frontage=round(frontage, 2))
            graph.add_edge(node_id, door_id,
                           weight=max(pt.distance(door_pt), 0.1), kind="room_to_door",
                           geometry=LineString([(pt.x, pt.y), (door_pt.x, door_pt.y)]))
            target, dist = nearest_spine(door_pt)
            if target is not None:
                graph.add_edge(door_id, target, weight=max(dist, 0.1),
                               kind="door_to_corridor",
                               geometry=LineString([(door_pt.x, door_pt.y),
                                                    graph.nodes[target]["pos"]]))
            stats["doors"] += 1
        else:
            # No corridor frontage. That is normal rather than broken: rooms
            # inside a suite, or a lab behind an ante-room, open into another
            # room and reach the corridor through it.
            #
            # The first version fell back to a straight line to the nearest
            # corridor centreline point, which is exactly the through-the-wall
            # behaviour this module exists to avoid. Instead, connect through
            # the shared boundary with the neighbouring room that this one
            # abuts most - still a real doorway on a real shared wall - and
            # let the route reach the corridor via that neighbour.
            best_neighbour = None
            for j, other in enumerate(others):
                if j == i:
                    continue
                found = find_door(room["poly"], other["poly"])
                if found is None:
                    continue
                door_pt, frontage = found
                if best_neighbour is None or frontage > best_neighbour[2]:
                    best_neighbour = (j, door_pt, frontage)

            if best_neighbour is not None:
                j, door_pt, frontage = best_neighbour
                graph.add_edge(node_id, f"{prefix}r{j}",
                               weight=max(pt.distance(door_pt) * 2, 0.5),
                               kind="room_to_room",
                               geometry=LineString([(pt.x, pt.y), (door_pt.x, door_pt.y)]))
                graph.nodes[node_id]["reaches_corridor_via_room"] = True
                stats["room_to_room"] += 1
            else:
                target, dist = nearest_spine(pt)
                if target is not None and dist is not None and dist <= MAX_FALLBACK_CONNECT_M:
                    graph.add_edge(node_id, target, weight=dist, kind="inferred_access",
                                   geometry=LineString([(pt.x, pt.y),
                                                        graph.nodes[target]["pos"]]))
                    graph.nodes[node_id]["door_inferred"] = True
                    stats["inferred_doors"] += 1

    return graph, stats, spaces


# --- verification ----------------------------------------------------------

WALL_CROSS_TOLERANCE_M = 0.45


def edges_through_walls(graph, spaces, circulation=None):
    """
    Find edges that leave the walkable space they belong to.

    The first version of this asked "does the edge pass through a space that
    isn't one of its endpoints", which sounds right and isn't: a corridor
    spine edge runs along the inside of corridor polygons, and those polygons
    are not formally its endpoints, so every spine edge was flagged. It
    reported over 100% of edges as violations, which is how it announced
    itself as a broken test rather than a broken graph.

    The property that actually matters is: **every edge stays inside walkable
    space it is entitled to be in.** For an edge that is:

      * part of the corridor spine - the circulation space
      * a room-to-door or door-to-corridor hop - that room plus circulation
      * a room-to-room hop - those two rooms

    Anything else it passes through is a wall crossing. The tolerance allows
    a little slack because room boundaries are drawn to the centre of the wall
    rather than its faces, so abutting polygons overlap by the wall thickness.
    """
    by_node = dict(spaces)
    violations = []
    for u, v, data in graph.edges(data=True):
        line = data.get("geometry")
        if line is None or line.is_empty or line.length <= 0:
            continue

        allowed = []
        if circulation is not None and not circulation.is_empty:
            allowed.append(circulation)
        for endpoint in (u, v):
            poly = by_node.get(endpoint)
            if poly is not None:
                allowed.append(poly)
        if not allowed:
            continue

        region = unary_union(allowed).buffer(WALL_CROSS_TOLERANCE_M)
        try:
            outside = line.difference(region)
        except Exception:
            continue
        stray = getattr(outside, "length", 0.0)
        if stray > WALL_CROSS_TOLERANCE_M:
            violations.append({
                "from": u, "to": v, "kind": data.get("kind"),
                "outside_m": round(stray, 2),
                "edge_m": round(line.length, 2),
            })
    return violations
