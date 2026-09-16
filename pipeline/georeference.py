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


def solve_building(floors, footprint):
    """
    floors: list of (floor_id, outline_polygon)
    Returns (rotation, {floor_id: params}).

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

    return rotation, ref_floor, results
