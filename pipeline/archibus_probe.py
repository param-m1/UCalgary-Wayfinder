"""
Work out how Archibus serves its floor-plan SVGs, so acquisition of all 518
floors can be done the cheap way instead of the fragile way.

Why this file exists
--------------------
The floor plans we need are rendered by Archibus's drawing viewer as inline
SVG. The obvious way to collect 518 of them is to drive the web UI: click a
building, click a floor, wait, scrape the DOM, repeat. That works, but it is
slow (seconds per floor), and every one of those 518 interactions is a chance
for a tree node to be missing, a click to land wrong, or a render to silently
not happen.

If the viewer fetches its drawing from a URL, we can skip the UI entirely and
pull all 518 over plain HTTP with the session cookies - far faster and with
far less to go wrong. That is worth ten minutes of looking before committing
to the clicking approach.

So this script logs in once (interactively - UCalgary uses SSO), opens one
known-good floor, and records every network request the page makes plus the
shape of the DOM it produces. It writes its findings to notes/probe-*.json
and prints a summary.

The one non-obvious trap, documented by whoever looked at this before: with a
headless or hidden browser the drawing silently fails to render - the
container stays empty while the page's own controller cheerfully reports the
floor as selected. So this runs headed, and waits on room paths actually
existing in the DOM rather than on a timer.

Session handling
----------------
Uses a persistent browser profile (data/browser-profile) rather than a
one-shot context, so the SSO session survives between runs and across
interruptions. Log in once; subsequent runs reuse it until it expires.
"""
import json
import os
import sys
import time

from playwright.sync_api import sync_playwright

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_DIR = os.path.join(PROJECT, "data", "browser-profile")
NOTES_DIR = os.path.join(PROJECT, "notes")

NAVIGATOR_URL = (
    "https://myfacilities.ucalgary.ca/archibus/schema/ab-core/views/"
    "navigator/ab-navigator.axvw"
)

# A floor confirmed to exist and render, used as the probe target.
PROBE_BUILDING = "SB"
PROBE_FLOOR = "01"

LOGIN_TIMEOUT_S = 600


def wait_for_login(page):
    """
    Blocks until we're actually inside Archibus rather than at the SSO wall.

    'Logged in' is detected by the Archibus app frame appearing, not by URL
    alone - the SSO flow bounces through several hosts and a URL check trips
    early on at least one of them.
    """
    print("\n" + "=" * 72)
    print("  A browser window has opened. Please sign in to UCalgary Archibus.")
    print("  Once the facilities page has finished loading, leave it open -")
    print("  this script detects the login automatically and carries on.")
    print("=" * 72 + "\n")

    deadline = time.time() + LOGIN_TIMEOUT_S
    while time.time() < deadline:
        try:
            url = page.url
            if "myfacilities.ucalgary.ca" in url and "login" not in url.lower():
                # the navigator view lives in the main document; its presence
                # means Archibus itself has served us a page
                if page.locator("body").count() and page.title():
                    content = page.content()
                    if "archibus" in content.lower() and "password" not in content.lower()[:4000]:
                        print(f"[login] detected Archibus session at {url}")
                        return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main():
    os.makedirs(PROFILE_DIR, exist_ok=True)
    os.makedirs(NOTES_DIR, exist_ok=True)

    requests_seen = []

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            PROFILE_DIR,
            headless=False,                      # see module docstring - headless does not render
            viewport={"width": 1600, "height": 1000},
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = context.pages[0] if context.pages else context.new_page()

        def on_request(req):
            requests_seen.append({
                "method": req.method,
                "url": req.url,
                "resource_type": req.resource_type,
                "post_data": (req.post_data or "")[:2000],
            })

        context.on("request", on_request)

        print(f"[probe] opening {NAVIGATOR_URL}")
        page.goto(NAVIGATOR_URL, wait_until="domcontentloaded", timeout=120000)

        if not wait_for_login(page):
            print("[probe] FAILED: no Archibus session detected within "
                  f"{LOGIN_TIMEOUT_S}s. Re-run and complete the sign-in.")
            context.close()
            return 1

        time.sleep(5)
        requests_seen.clear()   # from here on, only what the drawing does

        # ---- describe the page structure -------------------------------
        frames = []
        for f in page.frames:
            frames.append({"name": f.name, "url": f.url})
        print(f"\n[probe] {len(frames)} frames:")
        for f in frames:
            print(f"   - {f['name'] or '(main)'}: {f['url'][:120]}")

        structure = page.evaluate("""() => {
            const out = {ids: [], iframes: []};
            document.querySelectorAll('[id]').forEach(e => {
                if (out.ids.length < 400) out.ids.push(e.id);
            });
            document.querySelectorAll('iframe').forEach(f => {
                out.iframes.push({id: f.id, src: f.src});
            });
            return out;
        }""")
        print(f"\n[probe] {len(structure['iframes'])} iframes in main document:")
        for f in structure["iframes"]:
            print(f"   - id={f['id']!r} src={f['src'][:140]}")

        findings = {
            "navigator_url": NAVIGATOR_URL,
            "frames": frames,
            "main_ids_sample": structure["ids"][:200],
            "iframes": structure["iframes"],
            "requests_after_load": requests_seen[:400],
        }
        out_path = os.path.join(NOTES_DIR, "probe-structure.json")
        with open(out_path, "w") as fh:
            json.dump(findings, fh, indent=2)
        print(f"\n[probe] wrote {out_path}")

        print("\n[probe] Browser left open for 240s so the floor tree can be "
              "explored by hand if needed. Navigate to a floor plan now if the "
              "script did not - every network request is being recorded.")
        time.sleep(240)

        findings["requests_after_load"] = requests_seen[:2000]
        with open(out_path, "w") as fh:
            json.dump(findings, fh, indent=2)

        interesting = [r for r in requests_seen
                       if any(k in r["url"].lower()
                              for k in ("svg", "drawing", "dwg", "axvw", "jsp", "json"))]
        print(f"\n[probe] {len(requests_seen)} requests recorded, "
              f"{len(interesting)} look drawing-related:")
        for r in interesting[:60]:
            print(f"   {r['method']:5s} {r['url'][:150]}")
            if r["post_data"]:
                print(f"         POST: {r['post_data'][:300]}")

        with open(out_path, "w") as fh:
            json.dump(findings, fh, indent=2)
        print(f"\n[probe] final findings in {out_path}")
        context.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
