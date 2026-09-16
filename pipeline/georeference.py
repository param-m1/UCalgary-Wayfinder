"""
Turn a floor's page coordinates into real-world coordinates.

Why this file exists
--------------------
Room polygons come out of the PDFs in page points - arbitrary numbers that
mean nothing outside that sheet of paper. To route between buildings, to draw
routes on a map, and to link one floor to the floor above it, every floor has
to be placed on the earth: rotated to the building's true orientation, scaled
so a page point becomes a known number of metres, and shifted to the right
spot on campus.

The reference we fit against is UCalgary's public ArcGIS building-footprint
service, which gives each building's real outline in EPSG:3780 (an Alberta
metric grid, so distances come out in metres).

The assumption that turned out to be wrong
------------------------------------------
The obvious approach - and what the brief for this work assumed - is one
transform per building: all floors of a building share a CAD coordinate
system, so solve once and apply to every floor.

That is not how these drawings work. Archibus fits each floor to its own page,
so every floor has its own scale and offset. Measured on Art Building:

    floor 01   room-union bbox (213,82)-(661,445)   union area 120,091
    floor 02   room-union bbox (192,47)-(686,441)   union area 165,407
    floor 06   room-union bbox (256,46)-(613,331)   union area  82,938

Same building, same physical slab, quite different page geometry. A single
per-building transform would misplace most floors.

What is genuinely shared per building is **rotation**: a building has one
orientation relative to north, and every floor of it is drawn the same way up.
So this module splits the difference:

  * **Rotation is solved once per building**, using its best-fitting floor
    (the one whose outline matches the footprint most convincingly - normally
    a large lower floor that fills the whole slab).
  * **Scale and position are solved per floor**, with the building's rotation
    held fixed.

That keeps the per-building consistency the brief wanted, while respecting
how the drawings are actually laid out. It also gives a free check: if a
floor's independently best rotation disagrees wildly with its building's, that
floor is suspect and gets reported.

How the fit is scored, and two wrong turns on the way there
-----------------------------------------------------------
The score is **intersection-over-union** (IoU) between the floor's outline and
the building footprint, maximised over rotation and scale.

The first attempt scored *coverage* instead - overlap divided by the smaller
area - reasoning that upper floors are genuinely smaller than the ground
footprint and shouldn't be punished for not filling it. That is degenerate:
coverage is maximised by shrinking a floor until it sits comfortably inside
the footprint. Every floor of the Art Building scored a flawless 1.000 at
roughly 70% of its true size, and only a render made it obvious. IoU punishes
being too small as well as too large.

The second attempt fitted the convex hull of the rooms. A hull throws away the
notches and wings that actually pin a fit down, so several scales fitted about
equally well and the optimiser drifted to whatever bound it was handed - five
Art Building floors all reported an identical 0.2661 m/pt, the top of the
search range. Fitting the real merged outline fixed that.

Which leaves one real problem: a floor that genuinely looks nothing like the
footprint. A small roof structure fitted against the whole building outline
gets inflated until it fills it - Administration's 04RF came out at 0.5553
m/pt against ~0.19 for every other floor in that building, a 3x error. So:

  * Floors scoring at least TRUSTED_FIT against the footprint are believed
    about their own scale.
  * The median of those becomes the building's consensus scale.
  * Floors below the threshold keep that scale and are only *positioned* -
    slid over the best-fitting floor, which is where a penthouse or mezzanine
    physically sits: above the core, not centred on the whole slab.

That correction is recorded per floor (`rescaled_from`, `scale_source`) rather
than applied silently.

Note that page scale is usually but not always shared across a building's
floors. Aurora Hall's floors 01-09 all land within 0.180-0.196 m/pt; the Art
Building splits into two groups because its lower floors include the parkade
and are drawn at a different zoom to the tower above. Hence scale per floor
with a consensus fallback, rather than one scale per building.

What breaks if the assumptions don't hold
-----------------------------------------
A building with no well-fitting floor at all has no consensus to fall back on,
and every floor keeps its own possibly-inflated scale. Those buildings are
reported in the verification summary rather than silently trusted.
"""
import json
import math
import os
import urllib.parse
import urllib.request

import numpy as np
import pyproj
from shapely.affinity import rotate, scale as shapely_scale, translate
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

WORKING_CRS = "EPSG:3780"      # NAD83(CSRS) / Alberta 3TM 114W - metres
OUTPUT_CRS = "EPSG:4326"
_to_wgs84 = pyproj.Transformer.from_crs(WORKING_CRS, OUTPUT_CRS, always_xy=True)

FEATURE_SERVICE = ("https://services5.arcgis.com/oegdSE8JU1e7fUHu/arcgis/rest/"
                   "services/UC_Buildings/FeatureServer/0/query")

# Archibus renamed this building; ArcGIS still carries the old code.
BUILDING_CODE_ALIASES = {"TRW": "CWPH"}

# Any floor scoring below this is flagged for review rather than trusted.
GOOD_FIT = 0.80

# A floor scoring at least this against the footprint is believed about its
# own scale; below it the floor inherits the building consensus instead.
TRUSTED_FIT = 0.60

ROTATION_COARSE_STEP = 2.0     # degrees
ROTATION_FINE_RANGE = 3.0
ROTATION_FINE_STEP = 0.25
# Multipliers applied to a floor's own area-ratio scale estimate. The range
# is wide because a floor that is genuinely smaller than the ground footprint
# has an area ratio that under-estimates its true scale.
SCALE_SEARCH = [0.70, 0.80, 0.88, 0.94, 1.0, 1.06, 1.12, 1.20, 1.30, 1.45, 1.60]


def to_wgs84(x, y):
    lon, lat = _to_wgs84.transform(x, y)
    return lon, lat


def fetch_footprints(cache_path):
    """All building footprints in one request, cached to disk."""
    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            return json.load(fh)
    query = FEATURE_SERVICE + "?" + urllib.parse.urlencode({
        "where": "1=1",
        "outFields": "BLD_ID,BuildingName",
        "returnGeometry": "true",
        "outSR": "3780",
        "f": "json",
    })
    with urllib.request.urlopen(query, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out = {}
    for feat in data.get("features", []):
        code = (feat.get("attributes") or {}).get("BLD_ID")
        rings = (feat.get("geometry") or {}).get("rings")
        if not code or not rings:
            continue
        out[code] = {"rings": rings,
                     "name": feat["attributes"].get("BuildingName")}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as fh:
        json.dump(out, fh)
    return out


def rings_to_polygon(rings):
    parts = []
    for ring in rings:
        if len(ring) < 4:
            continue
        try:
            g = Polygon(ring)
            if not g.is_valid:
                g = g.buffer(0)
            if not g.is_empty:
                parts.append(g)
        except Exception:
            continue
    if not parts:
        return None
    merged = unary_union(parts)
    return merged if not merged.is_empty else None


def floor_outline(floor_data, simplify_pt=1.5):
    """
    The merged outline of a floor's rooms - the shape we fit to the footprint.

    This started out as the convex hull, on the grounds that hulls are cheap to
    intersect and shrug off a missing room. The problem is that a hull throws
    away exactly the information that pins a fit down: a building's notches,
    wings and re-entrant corners. With only a blobby hull to go on, several
    scales fit about equally well and the optimiser drifts to whatever bound it
    is given.

    Using the real merged outline (lightly simplified, since room boundaries
    carry far more vertices than the fit needs) keeps those shape features, so
    scale and rotation are much better determined.
    """
    polys = []
    for room in floor_data.get("rooms", []):
        ring = room.get("ring")
        if not ring or len(ring) < 4:
            continue
        try:
            g = Polygon(ring)
            if not g.is_valid:
                g = g.buffer(0)
            if not g.is_empty and g.geom_type == "Polygon":
                polys.append(g)
        except Exception:
            continue
    if not polys:
        return None
    merged = unary_union(polys)
    if merged.is_empty:
        return None
    # close hairline gaps between adjoining rooms so the union is one piece
    merged = merged.buffer(0.4).buffer(-0.4)
    if merged.is_empty:
        return None
    simplified = merged.simplify(simplify_pt)
    return simplified if not simplified.is_empty else merged


def page_polygons(floor_data):
    """Every room on a floor as a polygon, still in page coordinates."""
    polys = []
    for room in floor_data.get("rooms", []):
        ring = room.get("ring")
        if not ring or len(ring) < 4:
            continue
        try:
            g = Polygon(ring)
            if not g.is_valid:
                g = g.buffer(0)
        except Exception:
            continue
        if g.is_empty or g.geom_type != "Polygon":
            continue
        polys.append(g)
    return polys


def floor_level(floor_id):
    """
    A number for a floor id, so that 'the floor above' means something.

    B2 < B1 < G1 < 01 < 01M < 02. Parkade levels sit below the basements, and
    a mezzanine ('01M') half a level above the floor it is named after.
    """
    s = str(floor_id).upper()
    if s.startswith("B"):
        try:
            return -int(s[1:] or 1)
        except ValueError:
            return -1
    if s.startswith("G"):
        return 0
    if s.startswith("P"):
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
        return base + (0.5 if rest.startswith("M") else 0.0)
    return 999


def _iou(a, b):
    """
    Intersection over union.

    The first version of this scored 'coverage' - overlap over the smaller
    area - reasoning that upper floors are genuinely smaller than the ground
    footprint so shouldn't be punished for not filling it. That was wrong, and
    a render made it obvious: coverage is maximised by shrinking a floor until
    it sits comfortably inside the footprint, so every floor of the Art
    Building scored a perfect 1.000 at about 70% of its true size. IoU
    penalises being too small as well as too large, which is what we need.
    """
    try:
        inter = a.intersection(b).area
    except Exception:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _coverage(a, b):
    """Overlap over the smaller area - reported as a diagnostic, not optimised."""
    try:
        inter = a.intersection(b).area
    except Exception:
        return 0.0
    denom = min(a.area, b.area)
    return inter / denom if denom > 0 else 0.0


def apply_transform(geom, params):
    g = rotate(geom, params["rotation"], origin=params["page_center"])
    g = shapely_scale(g, xfact=params["scale"], yfact=params["scale"],
                      origin=params["page_center"])
    px, py = params["page_center"]
    rx, ry = params["real_center"]
    return translate(g, xoff=rx - px, yoff=ry - py)


def fit_floor(outline, footprint, rotations, scale_hint=None):
    """
    Find rotation / scale / translation placing `outline` onto `footprint`.

    Translation always aligns centroids; rotation is searched over the
    supplied candidates; scale starts from the area ratio and is searched
    around it (or around scale_hint when the building already has a consensus).
    """
    page_c = outline.centroid
    real_c = footprint.centroid
    base = {
        "page_center": (page_c.x, page_c.y),
        "real_center": (real_c.x, real_c.y),
    }
    # Seed from this floor's own area ratio. Seeding every floor from the
    # reference floor's scale pinned smaller floors against the top of the
    # search range, which showed up as several Art Building floors all
    # reporting an identical 0.2661 m/pt.
    centre = math.sqrt(footprint.area / outline.area) if outline.area > 0 else 1.0

    best = None
    for rot in rotations:
        for factor in SCALE_SEARCH:
            params = dict(base, rotation=rot, scale=centre * factor)
            score = _iou(apply_transform(outline, params), footprint)
            if best is None or score > best[0]:
                best = (score, rot, centre * factor)
    score, rot, sc = best
    params = dict(base, rotation=rot, scale=sc)
    placed = apply_transform(outline, params)
    params["score"] = score
    params["coverage"] = _coverage(placed, footprint)
    return params


def solve_building(floors, footprint, floor_polys=None):
    """
    floors: list of (floor_id, outline_polygon)
    floor_polys: optional {floor_id: [page-coordinate room polygons]}, which
        turns on the core-alignment pass that stacks floors on each other.
    Returns (rotation, reference_floor_id, {floor_id: params}).

    The building's rotation comes from whichever floor fits best on a full
    rotation sweep - normally a big lower floor. Every floor is then refit with
    that rotation held fixed, so floors can't drift out of alignment with each
    other.
    """
    coarse = [r * ROTATION_COARSE_STEP
              for r in range(int(360 / ROTATION_COARSE_STEP))]

    # 1. find the building's orientation from its most convincing floor
    best_ref = None
    for floor_id, outline in floors:
        fit = fit_floor(outline, footprint, coarse)
        if best_ref is None or fit["score"] > best_ref[1]["score"]:
            best_ref = (floor_id, fit)
    ref_floor, ref_fit = best_ref

    fine = [ref_fit["rotation"] + d * ROTATION_FINE_STEP
            for d in range(int(-ROTATION_FINE_RANGE / ROTATION_FINE_STEP),
                           int(ROTATION_FINE_RANGE / ROTATION_FINE_STEP) + 1)]
    ref_fit = fit_floor(dict(floors)[ref_floor], footprint, fine)
    rotation = ref_fit["rotation"]

    # 2. place every floor at that rotation, fitting scale independently
    results = {}
    for floor_id, outline in floors:
        fit = fit_floor(outline, footprint, [rotation])
        fit["trusted"] = fit["score"] >= TRUSTED_FIT
        results[floor_id] = fit

    # 3. rescue the floors whose own fit isn't credible
    #
    # A small roof structure or a one-room mezzanine looks nothing like the
    # building footprint, so fitting it against the footprint inflates it until
    # it fills the outline - Administration's 04RF came out at 0.5553 m/pt
    # against ~0.19 for every other floor in that building, a 3x error.
    #
    # Floors that clearly do match the footprint are the ones that can be
    # believed about scale, so their median scale becomes the building's. A
    # floor that doesn't match keeps that scale and is only positioned: it is
    # slid over the nearest trusted floor, which is where a penthouse or
    # mezzanine actually sits - above the core, not centred on the whole slab.
    trusted = [(fid, f) for fid, f in results.items() if f["trusted"]]
    if trusted:
        scales = sorted(f["scale"] for _, f in trusted)
        consensus = scales[len(scales) // 2]
        anchor_id, anchor = max(trusted, key=lambda t: t[1]["score"])
        anchor_outline = dict(floors)[anchor_id]
        anchor_placed = apply_transform(anchor_outline, anchor)

        for floor_id, outline in floors:
            fit = results[floor_id]
            if fit["trusted"]:
                continue
            page_c = outline.centroid
            rescued = {
                "page_center": (page_c.x, page_c.y),
                "real_center": (anchor_placed.centroid.x, anchor_placed.centroid.y),
                "rotation": rotation,
                "scale": consensus,
            }
            placed = apply_transform(outline, rescued)
            rescued["score"] = _iou(placed, footprint)
            rescued["coverage"] = _coverage(placed, footprint)
            rescued["overlap_with_anchor"] = _coverage(placed, anchor_placed)
            rescued["rescaled_from"] = round(fit["scale"], 4)
            rescued["trusted"] = False
            rescued["scale_source"] = f"building consensus (anchor {anchor_id})"
            results[floor_id] = rescued

    _align_floors_to_each_other(floors, results)
    if floor_polys:
        align_floors_by_cores(floor_polys, results)
        # the fit scores were measured before the floors were stacked on each
        # other, so refresh them against the footprint
        outlines = dict(floors)
        for floor_id, params in results.items():
            if "core_alignment" not in params or floor_id not in outlines:
                continue
            placed = apply_transform(outlines[floor_id], params)
            params["score"] = _iou(placed, footprint)
            params["coverage"] = _coverage(placed, footprint)
    return rotation, ref_floor, results


# How far a floor may be nudged to line up with the rest of its building.
# Searched coarse-then-fine: a flat grid at the fine step over this range would
# be thousands of polygon intersections per floor, and a range narrow enough to
# be affordable that way wasn't wide enough - CCIT's floors are more than 6 m
# out from each other and every one of them pinned against a +/-6 m bound.
MAX_FLOOR_NUDGE_M = 16.0
NUDGE_COARSE_STEP_M = 2.0
NUDGE_FINE_RANGE_M = 2.5
NUDGE_FINE_STEP_M = 0.5


def _align_floors_to_each_other(floors, results):
    """
    Nudge each floor so it lines up with the rest of its building.

    Fitting every floor independently against the building footprint gets each
    one roughly right, but leaves them a couple of metres out *relative to each
    other* - and that matters more than it sounds. Vertical links are found by
    looking for small spaces (stair and lift cores, a few metres across) that
    occupy the same place on consecutive floors. A 2-3 m relative offset drops
    their overlap below the matching threshold and no link is made at all.

    The symptom was stark: CCIT came out with **zero** vertical links and all
    seven floors as separate islands, so six of them were unreachable from the
    campus network no matter how good each floor's own fit was.

    So after the footprint fit, each floor is shifted by up to a few metres to
    maximise overlap with the building's best-fitting floor. Rotation and scale
    are left alone - those come from the footprint fit, which is the thing
    anchored to the real world; only the relative offset is being cleaned up.
    """
    if len(floors) < 2:
        return
    outlines = dict(floors)
    anchor_id = max(results, key=lambda f: results[f].get("score", 0))
    anchor = apply_transform(outlines[anchor_id], results[anchor_id])

    # Scored on IoU, not coverage, for the reason set out in _iou: coverage is
    # flat once the smaller floor is anywhere inside the bigger one, so it
    # cannot tell a floor that is lined up from one sitting three metres off.
    # Social Sciences showed exactly that - every floor "aligned", and every
    # stair core on floors 5 and 6 still 3.5 m apart.
    def search(placed, centre, span, step):
        best = (_iou(translate(placed, xoff=centre[0], yoff=centre[1]), anchor),
                centre[0], centre[1])
        count = int(span / step)
        for i in range(-count, count + 1):
            for j in range(-count, count + 1):
                dx, dy = centre[0] + i * step, centre[1] + j * step
                if abs(dx) > MAX_FLOOR_NUDGE_M or abs(dy) > MAX_FLOOR_NUDGE_M:
                    continue
                score = _iou(translate(placed, xoff=dx, yoff=dy), anchor)
                if score > best[0]:
                    best = (score, dx, dy)
        return best

    for floor_id, outline in floors:
        if floor_id == anchor_id:
            results[floor_id]["floor_nudge_m"] = [0.0, 0.0]
            continue
        params = results[floor_id]
        placed = apply_transform(outline, params)
        coarse = search(placed, (0.0, 0.0), MAX_FLOOR_NUDGE_M, NUDGE_COARSE_STEP_M)
        best = search(placed, (coarse[1], coarse[2]),
                      NUDGE_FINE_RANGE_M, NUDGE_FINE_STEP_M)
        _, dx, dy = best
        if dx or dy:
            rx, ry = params["real_center"]
            params["real_center"] = (rx + dx, ry + dy)
        params["floor_nudge_m"] = [round(dx, 2), round(dy, 2)]
        params["alignment_with_anchor"] = round(best[0], 3)


# --- lining floors up on the cores that run through them --------------------
#
# A small space that appears in the same place on two adjacent floors is a
# stair, a lift shaft, a duct or a stacked washroom - the parts of a building
# that genuinely repeat floor to floor. Matching those is a far sharper signal
# for "these two floors are lined up" than comparing outlines, because there
# are dozens of them scattered across the plan instead of one blobby shape.
#
# Small enough to be a core rather than a room, big enough not to be a stray
# fragment of geometry:
CORE_MAX_AREA_M2 = 90.0
CORE_MIN_AREA_M2 = 1.0

# A core counts as matched when its centre lands this close to a core of
# similar size on the other floor.
CORE_MATCH_RADIUS_M = 1.2
CORE_AREA_RATIO = 1.35

# How far the fit is allowed to be corrected. The scale range is wide because
# the errors being corrected are real and large: MacEwan Hall's floor 1 fits
# the footprint at 0.2024 m/pt and its floor 3 at 0.2723, a 35% disagreement
# between two floors of the same building, because each sheet is drawn to fill
# its own page and each was fitted to the whole footprint independently.
CORE_SCALE_RANGE = 0.22
CORE_SCALE_STEP = 0.02
CORE_MAX_SHIFT_M = 8.0
CORE_VOTE_BIN_M = 1.0

# Below this many matched cores the answer is noise, and the floor keeps the
# placement its own footprint fit gave it. Better an unlinked floor than a
# floor slid several metres on the strength of three coincidences.
CORE_MIN_MATCHES = 5


def _cores(polys):
    """Centres and areas of the small spaces on a placed floor."""
    pts, areas = [], []
    for p in polys:
        if CORE_MIN_AREA_M2 <= p.area <= CORE_MAX_AREA_M2:
            c = p.centroid
            pts.append((c.x, c.y))
            areas.append(p.area)
    return (np.asarray(pts, dtype=float).reshape(-1, 2),
            np.asarray(areas, dtype=float))


def _core_matches(ref_pts, ref_areas, pts, areas, centre, factor, dx, dy):
    """How many of this floor's cores land on a similar core of the reference."""
    if len(pts) == 0 or len(ref_pts) == 0:
        return 0
    moved = (pts - centre) * factor + centre + np.array([dx, dy])
    dist = np.hypot(moved[:, 0, None] - ref_pts[None, :, 0],
                    moved[:, 1, None] - ref_pts[None, :, 1])
    mine = (areas * factor * factor)[:, None]
    theirs = ref_areas[None, :]
    ratio = np.maximum(mine, theirs) / np.maximum(np.minimum(mine, theirs), 1e-6)
    return int(((dist <= CORE_MATCH_RADIUS_M) & (ratio <= CORE_AREA_RATIO))
               .any(axis=1).sum())


def _best_core_alignment(ref_polys, polys):
    """
    Find the scale factor and shift that best stacks this floor on the reference.

    Rather than search the (scale, dx, dy) space directly - tens of thousands of
    combinations per floor - each candidate scale lets every possible pairing of
    a core here with a core there *vote* for the shift that would bring them
    together. The shift the most pairs agree on is the answer, and only the few
    leading candidates are then scored properly. That turns the search from a
    grid sweep into two matrix operations per scale.

    Returns (matches, factor, dx, dy, centre) or None.
    """
    ref_pts, ref_areas = _cores(ref_polys)
    pts, areas = _cores(polys)
    if len(ref_pts) < CORE_MIN_MATCHES or len(pts) < CORE_MIN_MATCHES:
        return None
    centre = pts.mean(axis=0)

    base = _core_matches(ref_pts, ref_areas, pts, areas, centre, 1.0, 0.0, 0.0)
    best = (base, 1.0, 0.0, 0.0)

    steps = int(CORE_SCALE_RANGE / CORE_SCALE_STEP)
    for i in range(-steps, steps + 1):
        factor = 1.0 + i * CORE_SCALE_STEP
        moved = (pts - centre) * factor + centre
        mine = (areas * factor * factor)[:, None]
        theirs = ref_areas[None, :]
        ratio = np.maximum(mine, theirs) / np.maximum(np.minimum(mine, theirs), 1e-6)
        ox = ref_pts[None, :, 0] - moved[:, None, 0]
        oy = ref_pts[None, :, 1] - moved[:, None, 1]
        usable = ((ratio <= CORE_AREA_RATIO)
                  & (np.abs(ox) <= CORE_MAX_SHIFT_M)
                  & (np.abs(oy) <= CORE_MAX_SHIFT_M))
        if not usable.any():
            continue
        bins = np.stack([np.round(ox[usable] / CORE_VOTE_BIN_M),
                         np.round(oy[usable] / CORE_VOTE_BIN_M)], axis=1)
        keys, counts = np.unique(bins, axis=0, return_counts=True)
        for idx in np.argsort(counts)[-3:]:
            dx, dy = keys[idx] * CORE_VOTE_BIN_M
            score = _core_matches(ref_pts, ref_areas, pts, areas, centre,
                                  factor, dx, dy)
            if score > best[0]:
                best = (score, factor, dx, dy)

    # settle the answer at a finer step than the vote bins
    score, factor, dx, dy = best
    for fi in np.arange(factor - CORE_SCALE_STEP, factor + CORE_SCALE_STEP + 1e-9,
                        CORE_SCALE_STEP / 4):
        for ddx in np.arange(dx - CORE_VOTE_BIN_M, dx + CORE_VOTE_BIN_M + 1e-9, 0.25):
            for ddy in np.arange(dy - CORE_VOTE_BIN_M, dy + CORE_VOTE_BIN_M + 1e-9, 0.25):
                v = _core_matches(ref_pts, ref_areas, pts, areas, centre,
                                  fi, ddx, ddy)
                if v > score:
                    score, factor, dx, dy = v, float(fi), float(ddx), float(ddy)

    if score < CORE_MIN_MATCHES or score <= base:
        return None
    return score, factor, dx, dy, centre


def align_floors_by_cores(floor_polys, results):
    """
    Stack a building's floors on each other, working outward from the anchor.

    The footprint fit places every floor against the *building*, which is the
    only thing tied to the real world, but it has no way of knowing that floor 6
    sits directly on floor 5. Measured across the campus that left 164 pairs of
    adjacent floors with no vertical link at all - no stair core on one floor
    overlapped its own continuation on the next - which is why a quarter of all
    rooms could not be reached from their own building's entrance.

    So after the footprint fit, each floor is re-scaled and shifted to stack on
    the floor next to it, and the chain runs outward from the building's
    best-fitting floor, which keeps its footprint fit and therefore its tie to
    the real world.

    `floor_polys` is {floor_id: [page-coordinate room polygons]}. Records what
    it did in each floor's params under "core_alignment".
    """
    usable = [fid for fid in results if floor_polys.get(fid)]
    if len(usable) < 2:
        return
    anchor_id = max(usable, key=lambda f: results[f].get("score", 0))
    order = sorted(usable, key=floor_level)
    at = order.index(anchor_id)

    # outward from the anchor in both directions, so each floor is aligned to a
    # neighbour that has already been aligned itself
    sequence = ([(order[i], order[i - 1]) for i in range(at + 1, len(order))]
                + [(order[i], order[i + 1]) for i in range(at - 1, -1, -1)])

    placed = {anchor_id: [apply_transform(p, results[anchor_id])
                          for p in floor_polys[anchor_id]]}

    for floor_id, reference in sequence:
        params = results[floor_id]
        mine = [apply_transform(p, params) for p in floor_polys[floor_id]]
        found = _best_core_alignment(placed.get(reference, []), mine)
        if found:
            matches, factor, dx, dy, (cx, cy) = found
            rx, ry = params["real_center"]
            params["scale"] *= factor
            params["real_center"] = ((rx - cx) * factor + cx + dx,
                                     (ry - cy) * factor + cy + dy)
            params["core_alignment"] = {
                "reference_floor": reference,
                "scale_factor": round(factor, 3),
                "shift_m": [round(float(dx), 2), round(float(dy), 2)],
                "cores_matched": int(matches),
            }
            mine = [apply_transform(p, params) for p in floor_polys[floor_id]]
        placed[floor_id] = mine
