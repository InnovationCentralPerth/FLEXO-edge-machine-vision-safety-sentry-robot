"""Serve this machine's webcam over HTTP so a remote GPU box with no camera
attached (e.g. brannigan) can run the server/detectors against it.

Run on whichever machine has the physical webcam (this WSL2 laptop today):

    uv run python scripts/camera_relay.py

Then on the GPU box, point the main server at it over Tailscale:

    CAMERA_SOURCE=http://<this-machine>.tailnet.ts.net:8100/frame \\
        uv run uvicorn icp_safety_vlm.server.app:app --host 0.0.0.0 --port 8000

Deliberately a single-JPEG poll endpoint (GET /frame), not an MJPEG stream —
same reasoning as the main server's /frame vs /stream: simpler to get right
across a flaky Tailscale link than keeping a multipart stream alive, and the
remote Camera polls this every ~30ms anyway (see server/app.py's
Camera._loop_remote). Capture logic (MJPEG fourcc, WSL2 usbipd passthrough)
mirrors server/app.py's Camera — see README's "WSL2 webcam passthrough"
section if /dev/video* isn't showing up here.
"""

from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", "480"))
RELAY_PORT = int(os.environ.get("RELAY_PORT", "8100"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))

app = FastAPI(title="ICP Safety VLM camera relay")

_lock = threading.Lock()
_latest_jpeg: bytes | None = None
_cap: cv2.VideoCapture | None = None
_stop = threading.Event()


def _capture_loop() -> None:
    global _latest_jpeg
    while not _stop.is_set():
        assert _cap is not None
        ok, frame = _cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with _lock:
                _latest_jpeg = buf.tobytes()


@app.on_event("startup")
def _startup() -> None:
    global _cap
    _cap = cv2.VideoCapture(CAMERA_INDEX)
    # Force MJPEG (compressed) capture — see server/app.py's Camera.start()
    # docstring; usbipd's virtual USB bus doesn't reliably carry raw YUYV.
    _cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    _cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    threading.Thread(target=_capture_loop, daemon=True).start()


@app.on_event("shutdown")
def _shutdown() -> None:
    _stop.set()
    if _cap:
        _cap.release()


@app.get("/frame")
def frame() -> Response:
    with _lock:
        jpeg = _latest_jpeg
    if jpeg is None:
        return Response(status_code=204)
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/health")
def health() -> dict:
    with _lock:
        have_frame = _latest_jpeg is not None
    return {"status": "ok" if have_frame else "starting", "camera_index": CAMERA_INDEX}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=RELAY_PORT)
