#!/usr/bin/env python3
"""
live_page_server.py - serve the FLEXO live page and relay the sentry's camera frame and verdict.

Run it on the Jetson, next to the sentry server (the one on port 8010):

    python3 live_page_server.py
    -> open  http://<jetson>:8080/      (add ?backend=vlm_ollama_ministral if the sentry is not locked)

Python 3 standard library only; it works without internet. Keep flexo_sentry_live.html in the
same folder as this file.

Why it exists: the page and the sentry are on different ports, so the browser would refuse to
read the sentry's verdict across them. This script serves the page and passes two things
through from the same address, so the browser sees one origin:

    /api/status  ->  <sentry>/status   (the verdict, as JSON)
    /api/frame   ->  <sentry>/frame    (the camera frame, as a JPEG)

It is not an open proxy. It only ever contacts the sentry address given with --sentry, only
those two paths, and the only thing a visitor can choose is a backend name made of letters,
digits, "_" and "-". Like the sentry's own dashboard it has no login: anyone who can reach this
port can see the camera, so keep it on your own network.
"""

import argparse
import http.server
import json
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

BACKEND_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_BYTES = 5 * 1024 * 1024
PAGE = pathlib.Path(__file__).resolve().with_name("flexo_sentry_live.html")


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "FlexoLive/1.0"
    sentry = "http://127.0.0.1:8010"

    def log_message(self, fmt, *args):          # keep the console quiet; errors are printed below
        pass

    def _send(self, code, body, ctype):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass                                # the browser went away mid-reply

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        if url.path in ("/", "/index.html"):
            try:
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            except OSError:
                return self._send(500, f"{PAGE.name} was not found next to this script.".encode(), "text/plain")
        if url.path == "/api/status":
            return self._relay("/status", url.query, want="json")
        if url.path == "/api/frame":
            return self._relay("/frame", url.query, want="image")
        self._send(404, b"Not found", "text/plain")

    def _relay(self, upstream_path, query, want):
        target = self.sentry + upstream_path
        backend = urllib.parse.parse_qs(query).get("backend", [""])[0]
        if backend:
            if not BACKEND_RE.match(backend):
                return self._json(400, {"status": "ERROR", "message": "Bad backend name."})
            target += "?backend=" + urllib.parse.quote(backend)
        try:
            with urllib.request.urlopen(target, timeout=4) as r:
                body = r.read(MAX_BYTES + 1)
                ctype = r.headers.get_content_type()
        except Exception as e:                  # the sentry is down, slow, or answered with an error
            if want == "json":
                return self._json(502, {"status": "OFFLINE", "message": f"Cannot reach the sentry ({type(e).__name__})."})
            return self._send(502, b"Cannot reach the sentry", "text/plain")
        if len(body) > MAX_BYTES:
            return self._send(502, b"Reply too large", "text/plain")
        if want == "json":
            try:
                data = json.loads(body)         # re-emit parsed JSON, never raw upstream bytes
            except ValueError:
                return self._json(502, {"status": "ERROR", "message": "The sentry sent something that is not JSON."})
            return self._json(200, data)
        if not ctype.startswith("image/"):
            return self._send(502, b"Not an image", "text/plain")
        return self._send(200, body, ctype)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Serve the FLEXO live page and relay the sentry's frame and verdict.")
    ap.add_argument("--sentry", default="http://127.0.0.1:8010", help="where the sentry server listens")
    ap.add_argument("--host", default="0.0.0.0", help="address to listen on (127.0.0.1 = this Jetson only)")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args(argv)
    if urllib.parse.urlsplit(args.sentry).scheme != "http":
        sys.exit("--sentry must be an http:// address, e.g. http://127.0.0.1:8010")
    Handler.sentry = args.sentry.rstrip("/")
    if not PAGE.exists():
        sys.exit(f"Missing {PAGE.name}: keep it in the same folder as this script.")
    srv = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    print(f"Live page on http://{args.host}:{args.port}/   (sentry: {Handler.sentry})   Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
