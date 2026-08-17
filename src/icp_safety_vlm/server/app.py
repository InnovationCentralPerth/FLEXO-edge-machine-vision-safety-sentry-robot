"""FastAPI app: webcam -> selected detector backend -> MJPEG stream + GO!/STOP! UI.

Run with:
    uv run uvicorn icp_safety_vlm.server.app:app --reload

Switch backend via env var or query string:
    DETECTOR_BACKEND=yolo_hsv uv run uvicorn icp_safety_vlm.server.app:app
    http://localhost:8000/?backend=vlm_qwen2vl
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from icp_safety_vlm.detectors import DetectionResult, get_detector, list_detectors

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="ICP Safety VLM Sentry")

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", "480"))
DEFAULT_BACKEND = os.environ.get("DETECTOR_BACKEND", "yolo_hsv")


class Camera:
    """Owns the single VideoCapture handle. V4L2/UVC webcams generally only
    allow one open handle at a time, so every backend's Sentry reads frames
    from here rather than opening the device itself — that's what caused
    the "can't open camera by index" failures when switching backends."""

    def __init__(self):
        self.cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        # Force MJPEG (compressed) capture. Raw YUYV needs far more USB
        # bandwidth than usbipd's virtual bus reliably provides under WSL2,
        # which corrupts frames into large frozen solid-color blocks.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap:
            self.cap.release()

    def _loop(self) -> None:
        while not self._stop.is_set():
            assert self.cap is not None
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            with self._lock:
                self._latest_frame = frame

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()


_camera = Camera()


IDLE_TIMEOUT_S = 8.0  # pause a backend's inference loop after this long unpolled


class Sentry:
    """Owns one detector backend, runs inference in a background thread
    (pulling frames from the shared `Camera`) so the MJPEG stream doesn't
    block on (potentially slow) inference, and exposes the latest annotated
    frame + verdict for the UI.

    Every backend the dropdown has ever selected stays alive in its own
    Sentry (see `get_sentry`). Two things keep multiple resident backends
    from starving each other on an 8GB card:

    1. The inference loop pauses itself once nobody has polled this Sentry
       recently (`latest_jpeg`/`latest_status`, i.e. the browser has
       navigated to a different backend) and resumes the moment it's
       polled again — otherwise every ever-visited backend keeps running
       flat-out forever, and N backends all doing that is N-way GPU
       scheduling contention (measured: turned a ~550ms/frame model into
       ~11s/frame with nothing else even changing).
    2. Pausing the *loop* doesn't free the *memory* — a multi-GB VLM stays
       resident in VRAM the whole time it's paused. Left long enough, an
       idle backend now calls `detector.unload()` to actually free that
       memory, and reloads via `warmup()` on the next poll (a few seconds,
       same cost as first-time startup). Otherwise two VLMs' worth of
       weights sitting in an 8GB card leaves so little headroom for the
       *active* one's own working buffers that it slows down too — from
       memory pressure, not scheduling contention, and idle-pausing the
       loop alone doesn't touch it."""

    def __init__(self, backend: str):
        self.backend_name = backend
        self.detector = get_detector(backend)
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_result: DetectionResult | None = None
        self._result_at: float = 0.0  # time.time() when _latest_result was produced
        self._error: str | None = None
        self._last_polled: float = time.time()
        self._loaded = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                idle = (time.time() - self._last_polled) > IDLE_TIMEOUT_S

            if idle:
                if self._loaded:
                    try:
                        self.detector.unload()
                    except Exception:  # noqa: BLE001 — unload best-effort, never fatal
                        pass
                    with self._lock:
                        self._loaded = False
                        self._latest_result = None
                        self._latest_jpeg = None
                time.sleep(0.2)
                continue

            if not self._loaded:
                try:
                    self.detector.warmup()
                    with self._lock:
                        self._loaded = True
                        self._error = None
                except Exception as exc:  # noqa: BLE001 — surface to the UI, not a 500
                    with self._lock:
                        self._error = f"{type(exc).__name__}: {exc}"
                    time.sleep(1.0)
                    continue

            frame = _camera.latest_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            try:
                result = self.detector.timed_infer(frame)
                annotated = _annotate(frame, result)
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            except Exception as exc:  # noqa: BLE001 — one bad frame shouldn't kill the loop
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.5)
                continue
            if ok:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()
                    self._latest_result = result
                    self._result_at = time.time()
                    self._error = None

    def debug_info(self) -> dict:
        """Read-only introspection — deliberately does NOT touch _last_polled,
        so checking this doesn't itself keep the backend un-idle."""
        with self._lock:
            return {
                "backend": self.backend_name,
                "idle_for_s": round(time.time() - self._last_polled, 1),
                "is_idle": (time.time() - self._last_polled) > IDLE_TIMEOUT_S,
                "loaded": self._loaded,
                "thread_alive": self._thread.is_alive() if self._thread else False,
            }

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            self._last_polled = time.time()
            return self._latest_jpeg

    def latest_status(self) -> dict:
        with self._lock:
            self._last_polled = time.time()
            r = self._latest_result
            result_at = self._result_at
            error = self._error
        if error is not None:
            return {
                "status": "ERROR",
                "backend": self.backend_name,
                "latency_ms": 0,
                "n_people": 0,
                "age_ms": 0,
                "message": error,
            }
        if r is None:
            return {"status": "STARTING", "backend": self.backend_name, "latency_ms": 0, "n_people": 0, "age_ms": 0}
        return {
            "status": r.status,
            "backend": r.backend,
            "latency_ms": round(r.latency_ms, 1),
            "n_people": len(r.detections),
            "message": r.message,
            # How stale this verdict is: time since the analyzed frame was
            # captured. Slow backends (VLM) update the on-screen frame in
            # bursts, not truly live — this makes that visible instead of
            # letting a STOP banner silently sit next to a scene that's
            # already changed.
            "age_ms": round((time.time() - result_at) * 1000, 0),
        }


def _annotate(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    for d in result.detections:
        x1, y1, x2, y2 = d.bbox
        # Whole-frame synthetic boxes (VLM backends without real per-person
        # localization) aren't worth drawing a rectangle for — they'd just
        # stack identical labels on top of each other.
        if (x2 - x1) >= 0.98 * w and (y2 - y1) >= 0.98 * h:
            continue
        color = (0, 200, 0) if d.wearing_vest else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = "VEST" if d.wearing_vest else "NO VEST"
        cv2.putText(out, label, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    banner_color = (0, 200, 0) if result.status == "GO" else (0, 0, 255)
    cv2.putText(out, result.status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, banner_color, 2)

    if result.message:
        _draw_wrapped_text(out, result.message, banner_color)
    return out


def _draw_wrapped_text(
    img: np.ndarray, text: str, color: tuple[int, int, int], max_width_frac: float = 0.94
) -> None:
    """Draw `text` word-wrapped, anchored to the bottom edge of the frame
    (grows upward with line count so it never overflows), with a
    translucent backing band so it stays legible over any background."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 1
    max_width = int(w * max_width_frac)

    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        (tw, _), _ = cv2.getTextSize(candidate, font, scale, thickness)
        if tw > max_width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    line_height = 22
    band_height = min(len(lines) * line_height + 12, h)
    band_top = h - band_height
    overlay = img.copy()
    cv2.rectangle(overlay, (0, band_top), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)

    y = band_top + line_height
    for line in lines:
        cv2.putText(img, line, (10, y), font, scale, color, thickness, cv2.LINE_AA)
        y += line_height


_sentries: dict[str, Sentry] = {}
_sentries_lock = threading.Lock()


def get_sentry(backend: str) -> Sentry:
    with _sentries_lock:
        if backend not in _sentries:
            s = Sentry(backend)
            s.start()
            _sentries[backend] = s
        return _sentries[backend]


@app.on_event("startup")
def _startup() -> None:
    _camera.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    for s in _sentries.values():
        s.stop()
    _camera.stop()


@app.get("/", response_class=HTMLResponse)
def index(request: Request, backend: str = Query(default=DEFAULT_BACKEND)):
    get_sentry(backend)  # ensure it's started
    return templates.TemplateResponse(
        request,
        "index.html",
        {"backend": backend, "available": list_detectors()},
    )


@app.get("/stream")
def stream(backend: str = Query(default=DEFAULT_BACKEND)):
    sentry = get_sentry(backend)

    def gen():
        boundary = b"--frame"
        while True:
            jpeg = sentry.latest_jpeg()
            if jpeg is not None:
                yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.03)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/frame")
def frame(backend: str = Query(default=DEFAULT_BACKEND)):
    """Single-JPEG snapshot, meant to be polled with a cache-busting query
    param. More robust than MJPEG (/stream) across browsers when switching
    between backends mid-stream, and the natural fit for a slow backend
    (VLM) that only produces a new frame every few seconds anyway — true
    30fps streaming buys nothing there."""
    sentry = get_sentry(backend)
    jpeg = sentry.latest_jpeg()
    if jpeg is None:
        return Response(status_code=204)
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/status")
def status(backend: str = Query(default=DEFAULT_BACKEND)):
    return get_sentry(backend).latest_status()


@app.get("/debug")
def debug():
    """Idle state of every currently-instantiated Sentry — `is_idle: true`
    means that backend's inference loop is paused (not consuming GPU) and
    `idle_for_s` is how long since its last poll. Useful for confirming
    which backend(s) are actually contending for the GPU at any moment;
    see the "Server architecture" section of README.md for why this
    exists — three backends idle-pausing incorrectly (or not at all) turned
    a ~550ms/frame model into ~11s/frame purely from contention once."""
    with _sentries_lock:
        return [s.debug_info() for s in _sentries.values()]
