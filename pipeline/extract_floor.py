"""
Pull room polygons, room numbers and wall geometry out of one Archibus floor
plan PDF.

Why this file exists
--------------------
Routing room-to-room needs to know what shape each room is and which rooms
touch which. The obvious reading of these PDFs says that information isn't
there: a floor is drawn as tens of thousands of disconnected stroked
segments, and a naive scan finds essentially no closed shapes. Reconstructing
rooms from that line soup - working out where segments cross and which
regions they enclose - is slow, fragile, and is where the previous attempt at
this project got stuck.

The room polygons are in fact present, in a layer that is easy to miss.

How they were found
-------------------
Every stroke in these PDFs carries a linewidth and a colour, and grouping by
those two values splits a drawing into its CAD layers. Doing that on
SB - 01 and rendering each group showed furniture in one group, text in
another, walls in a third - and one small group of paths with **linewidth 0**
(a hairline) that turned out to be the room boundaries.

The decisive property: *every* path in that layer is closed. Not most - all
of them, on every floor checked:

    SB - 01   107 paths / 107 closed     AU - 03    95 /  95
    CR - 04   147 paths / 147 closed     SS - 10    61 /  61
    ENG - 04   37 paths /  37 closed     LSIH - 03  65 /  65

Across 18 randomly sampled floors, 488 of 502 room-number labels (97%) fall
inside one of these polygons. So each closed hairline path is a room, and the
room number is the text sitting inside it.

This matters beyond convenience. Having true room polygons means adjacent
rooms share a boundary, and a room's door must lie on the boundary it shares
with a corridor. That is what lets the router connect a room to the corridor
network *through its doorway* instead of drawing a straight line out of the
middle of the room and through whatever walls are in the way - the specific
bug this project exists to fix.

Two things must be filtered out
-------------------------------
1. **The page frame and title block.** Two polygons appear in every single
   drawing at identical coordinates - area 337,394 at (36,36)-(806,474) and
   area 65,471 at (36,474)-(806,559). They're the sheet border and the title
   block, not rooms. Rejected by size relative to the page.

2. **Off-page garbage text.** Some drawings carry encoded strings rendered as
   text far outside the page - ENA - B2 has tokens like "2662720185341" and
   "0000000" sitting at y = -304. They match a room-number pattern but are
   not room numbers. Rejected by requiring the text to sit inside the page.

What breaks if the assumptions don't hold
-----------------------------------------
If a drawing has no linewidth-0 layer, this yields zero rooms and the floor is
reported as empty rather than guessed at. Some floors genuinely are nearly
empty - roof levels and mezzanines especially (MTH - 06RF has labels but no
room polygons at all). Those are recorded with what they have and flagged, not
padded out with invented geometry.

Coordinates here are PDF page points with y already flipped to y-up, so the
output is in a normal mathematical orientation. Converting to real-world
coordinates happens later, in georeference.py.
"""
import json
import os
import re
import warnings

import pdfplumber
from shapely.geometry import Point, Polygon

warnings.filterwarnings("ignore")

# Room numbers look like "148", "148A", "199Z", "0101RF". A bare one or two
# digits is usually a floor tag or a stray dimension, so require at least two
# characters of digits before any suffix.
ROOM_NUMBER_RE = re.compile(r"^\d{2,5}[A-Za-z]{0,3}$")

# The sheet border and title block are the two largest closed paths and are
# present in every drawing. Anything covering more than this fraction of the
# page is furniture of the drawing itself, not a room.
MAX_ROOM_PAGE_FRACTION = 0.35

# The title block is only 13% of the page by area, so it slips past the
# fraction test above - but it is a 770x85 strip spanning 91% of the page
# width, and no real room does that. Caught by extent instead of area.
# (Left unchecked it becomes the largest "room" on all 518 floors, and steals
# the floor-id label - "01" - as its room number.)
MAX_ROOM_EXTENT_FRACTION = 0.85

# Below this a "polygon" is a sliver artifact rather than a space.
MIN_ROOM_AREA_PT2 = 4.0

HAIRLINE = 0.0


def _closed(pts, tol=0.05):
    return (len(pts) >= 4
            and abs(pts[0][0] - pts[-1][0]) < tol
            and abs(pts[0][1] - pts[-1][1]) < tol)


def extract_floor(pdf_path):
    """
    Returns a dict describing one floor in page coordinates (y-up):

        {
          "page": [width, height],
          "rooms": [ {"number": "148" | None, "ring": [[x, y], ...]}, ... ],
          "wall_path_count": int,
          "unmatched_labels": [ {"text": ..., "x":, "y":}, ... ],
          "stats": {...}
        }

    Raw wall strokes are deliberately NOT stored. They were at first, and it
    cost ~200 MB across 518 floors for data that turns out to be the wrong
    barrier to test against: the non-hairline strokes include all the
    furniture and fixtures, so every routing edge inside a room "crosses" a
    chair. The meaningful barrier is the room boundary itself, which the room
    polygons already give us. Only the stroke count is kept, as a record.
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        width, height = page.width, page.height
        page_area = width * height

        rings = []
        wall_path_count = 0
        for item in list(page.curves) + list(page.lines):
            lw = item.get("linewidth")
            pts = item.get("pts")
            if pts:
                flipped = [(x, height - y) for x, y in pts]
            elif "x0" in item:
                flipped = [(item["x0"], height - item["y0"]),
                           (item["x1"], height - item["y1"])]
            else:
                continue

            if lw is not None and lw == HAIRLINE:
                if _closed(flipped):
                    rings.append(flipped)
            elif len(flipped) >= 2:
                wall_path_count += 1

        labels = []
        for w in page.extract_words():
            text = w["text"]
            if not ROOM_NUMBER_RE.match(text):
                continue
            x = (w["x0"] + w["x1"]) / 2
            y = height - (w["top"] + w["bottom"]) / 2
            # off-page garbage (see module docstring)
            if not (0 <= x <= width and 0 <= y <= height):
                continue
            labels.append({"text": text, "x": x, "y": y})

    polys, kept_rings = [], []
    frame_rejects = 0
    for ring in rings:
        try:
            g = Polygon(ring)
            if not g.is_valid:
                g = g.buffer(0)
        except Exception:
            continue
        if g.is_empty or g.geom_type != "Polygon":
            continue
        if g.area < MIN_ROOM_AREA_PT2:
            continue
        if g.area > page_area * MAX_ROOM_PAGE_FRACTION:
            frame_rejects += 1
            continue
        minx, miny, maxx, maxy = g.bounds
        if ((maxx - minx) > width * MAX_ROOM_EXTENT_FRACTION
                or (maxy - miny) > height * MAX_ROOM_EXTENT_FRACTION):
            frame_rejects += 1
            continue
        polys.append(g)
        kept_rings.append(list(g.exterior.coords))

    numbers = assign_labels(polys, labels)

    matched_label_ids = set()
    rooms = []
    for i, ring in enumerate(kept_rings):
        rooms.append({"number": numbers.get(i), "ring": [[round(x, 3), round(y, 3)]
                                                          for x, y in ring]})
    for idx in numbers.values():
        pass
    used_text = {v for v in numbers.values() if v}
    unmatched = [l for l in labels if l["text"] not in used_text]

    return {
        "page": [width, height],
        "rooms": rooms,
        "unmatched_labels": unmatched,
        "stats": {
            "closed_hairline_paths": len(rings),
            "rooms_kept": len(rooms),
            "rooms_numbered": sum(1 for r in rooms if r["number"]),
            "frame_polygons_rejected": frame_rejects,
            "labels_found": len(labels),
            "labels_unmatched": len(unmatched),
            "wall_paths": wall_path_count,
        },
    }


def assign_labels(polys, labels):
    """
    Map each room polygon index to its room number.

    Carried over from the previous pipeline, where this was learned the hard
    way: assignment has to be global rather than per-polygon. A label point can
    sit inside several nested polygons, so each label goes to the *smallest*
    polygon containing it and is then used up. Doing it per-polygon copies one
    number onto every polygon that encloses it, and the same room number shows
    up several times on one floor.
    """
    candidates = []
    for label in labels:
        pt = Point(label["x"], label["y"])
        containing = [i for i, g in enumerate(polys) if g.contains(pt)]
        if not containing:
            continue
        best = min(containing, key=lambda i: polys[i].area)
        candidates.append((polys[best].area, best, label["text"]))

    assigned, used = {}, set()
    for _, index, text in sorted(candidates):
        if index in assigned or text in used:
            continue
        assigned[index] = text
        used.add(text)
    return assigned


def main():
    import argparse
    import csv
    import sys
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="/Users/parammehta/Downloads/UCalgary_Archibus_FloorPlans")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="limit to one building code")
    args = ap.parse_args()

    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out or os.path.join(project, "data", "floors")
    os.makedirs(out_dir, exist_ok=True)

    rows = list(csv.DictReader(open(os.path.join(args.source, "MANIFEST.csv"))))
    if args.only:
        rows = [r for r in rows if r["building_code"] == args.only]

    started = time.time()
    summary = []
    for n, row in enumerate(rows, 1):
        code, floor = row["building_code"], row["floor_id"]
        pdf_path = os.path.join(args.source, row["relative_path"])
        target_dir = os.path.join(out_dir, code)
        os.makedirs(target_dir, exist_ok=True)
        target = os.path.join(target_dir, f"{code}-{floor}.json")

        if os.path.exists(target):          # checkpointed: resume cheaply
            continue
        try:
            data = extract_floor(pdf_path)
        except Exception as e:
            print(f"[{n}/{len(rows)}] {code} {floor}: FAILED {type(e).__name__}: {e}")
            summary.append({"code": code, "floor": floor, "error": str(e)[:200]})
            continue

        data["building_code"] = code
        data["floor_id"] = floor
        data["source_pdf"] = row["relative_path"]
        with open(target, "w") as fh:
            json.dump(data, fh)

        s = data["stats"]
        summary.append({"code": code, "floor": floor, **s})
        if n % 25 == 0 or s["rooms_kept"] == 0:
            print(f"[{n}/{len(rows)}] {code} {floor}: rooms={s['rooms_kept']} "
                  f"numbered={s['rooms_numbered']} unmatched={s['labels_unmatched']}")
            sys.stdout.flush()

    with open(os.path.join(project, "data", "extract-summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\ndone in {time.time()-started:.0f}s; {len(summary)} floors processed this run")


if __name__ == "__main__":
    main()
