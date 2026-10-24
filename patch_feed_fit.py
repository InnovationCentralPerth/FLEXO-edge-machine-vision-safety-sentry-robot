#!/usr/bin/env python3
"""
patch_feed_fit.py - make the camera picture on the sentry's dashboard fill the screen.

    python3 patch_feed_fit.py                 apply (safe to run twice)
    python3 patch_feed_fit.py --compact       also hide the title and the backend drop-down, for even more picture
    python3 patch_feed_fit.py --all-screens   apply on wide screens too (by default only on tall, portrait ones)
    python3 patch_feed_fit.py --undo          remove it again
    python3 patch_feed_fit.py --dry-run       show what would change, change nothing

The dashboard is the template  ~/projects/icp_safety_vlm/src/icp_safety_vlm/server/templates/index.html  (override with
--template). This adds ONE marked block of CSS just before </head>. It turns the page into a full-height column and lets the
camera picture (<img id="feed">) take all the height that the title, banner and text leave over, scaled to fit without
stretching. Nothing else in the file is touched, and the first run saves the original as index.html.bak-before-feedfit.

By default the layout applies only when the screen is taller than it is wide (the portrait LCD). On a wide, short screen
(a laptop, a landscape monitor) the page stays exactly as it was, because there the title, banner and text already use most of
the height and a fit-to-screen picture would come out smaller, not bigger.

After applying, reload the page in the browser (F5). The template is read afresh on each request, so nothing needs restarting;
if the old layout still shows, restart the sentry.
"""
import argparse
import os
import re
import shutil
import sys
import tempfile

DEFAULT_TEMPLATE = "~/projects/icp_safety_vlm/src/icp_safety_vlm/server/templates/index.html"
BEGIN = "<!-- flexo-fit:begin (made by patch_feed_fit.py; remove with --undo) -->"
END = "<!-- flexo-fit:end -->"
BLOCK_RE = re.compile(re.escape(BEGIN.split(" (")[0]) + r".*?" + re.escape(END) + r"\n?", re.S)

CSS = """\
html { height: 100% !important; }
body {
  box-sizing: border-box !important; margin: 0 !important; padding: 6px 8px !important;
  height: 100vh !important; height: 100dvh !important; min-height: 0 !important;
  display: flex !important; flex-direction: column !important; align-items: center !important; justify-content: flex-start !important;
  gap: 4px !important; overflow-x: hidden !important; overflow-y: auto !important;
}
body > * { flex: 0 0 auto; }
h1 { margin: 0 !important; font-size: 1.15rem !important; }
#banner { margin-top: 0 !important; margin-bottom: 0 !important; }
#message, #meta { margin-top: 0 !important; margin-bottom: 0 !important; }
#feed {
  flex: 1 1 0 !important; min-height: 120px !important; height: auto !important;
  width: 100% !important; max-width: none !important; margin: 0 !important;
  object-fit: contain !important; border: 0 !important; outline: 0 !important; background: transparent !important;
}
"""
COMPACT_CSS = "h1, body > div:first-of-type { display: none !important; }\n"


def block(compact: bool, all_screens: bool = False) -> str:
    css = CSS + (COMPACT_CSS if compact else "")
    if not all_screens:
        css = "@media (orientation: portrait) {\n" + "".join("  " + l + "\n" for l in css.splitlines()) + "}\n"
    for bad in ("{{", "{%", "{#"):                        # the template is rendered by Jinja: these would be read as template code
        assert bad not in css, f"the CSS contains {bad}"
    return f"{BEGIN}\n<style id=\"flexo-fit\">\n{css}</style>\n{END}\n"


def remove_block(text: str) -> str:
    return BLOCK_RE.sub("", text)


def apply(text: str, compact: bool, all_screens: bool = False) -> str:
    text = remove_block(text)
    m = re.search(r"</head>", text, re.I)
    if not m:
        raise SystemExit("This does not look like the dashboard template: there is no </head>. Nothing changed.")
    if 'id="feed"' not in text and "id='feed'" not in text:
        raise SystemExit("This does not look like the dashboard template: there is no element with id=\"feed\" (the camera picture). Nothing changed.")
    return text[:m.start()] + block(compact, all_screens) + text[m.start():]


def write_atomic(path: str, text: str) -> None:
    st = os.stat(path)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".index-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.chmod(tmp, st.st_mode & 0o7777)
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Make the dashboard's camera picture fill the screen.")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE, help="the dashboard template (default: the sentry project's index.html)")
    ap.add_argument("--compact", action="store_true", help="also hide the title and the backend drop-down")
    ap.add_argument("--all-screens", action="store_true", help="apply on wide screens too, not only in portrait")
    ap.add_argument("--undo", action="store_true", help="remove the block again")
    ap.add_argument("--dry-run", action="store_true", help="show what would change, change nothing")
    args = ap.parse_args(argv)

    path = os.path.expanduser(args.template)
    if not os.path.isfile(path):
        raise SystemExit(f"Not found: {path}\nGive the right file with --template.")
    with open(path, encoding="utf-8", newline="") as f:
        old = f.read()

    if args.undo:
        new = remove_block(old)
        what = "removed" if new != old else "nothing to remove (not applied)"
    else:
        new = apply(old, args.compact, args.all_screens)
        what = "applied" + (" (compact)" if args.compact else "") + (" on all screens" if args.all_screens else " (portrait screens)")
    if new == old:
        print(f"{path}: {what}; already as wanted.")
        return 0
    if args.dry_run:
        print(f"{path}: would be {what}. Lines {old.count(chr(10)) + 1} -> {new.count(chr(10)) + 1}. Nothing changed (--dry-run).")
        return 0

    backup = path + ".bak-before-feedfit"
    if not os.path.exists(backup) and not args.undo:
        shutil.copy2(path, backup)
        print(f"Saved the original as {backup}")
    write_atomic(path, new)
    print(f"{path}: {what}.")
    if not args.undo:
        print("Reload the page in the browser (F5). To go back:  python3 patch_feed_fit.py --undo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
