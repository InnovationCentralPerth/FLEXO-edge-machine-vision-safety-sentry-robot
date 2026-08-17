"""Shared interface every detector backend (classic CV/YOLO, VLM, ...) implements.

Keeping this contract narrow is what lets the FastAPI server, the benchmark
script, and any future backend (e.g. a fine-tuned edge VLM for the Pi 5) stay
decoupled from *how* a given backend decides GO!/STOP!.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """One detected person and their vest status."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coords
    wearing_vest: bool
    confidence: float  # 0..1, backend-defined meaning (detector score / VLM certainty proxy)


@dataclass
class DetectionResult:
    """Full-frame verdict returned by a detector for one image."""

    detections: list[Detection] = field(default_factory=list)
    latency_ms: float = 0.0
    backend: str = ""
    # Optional human-readable, backend-authored message (e.g. a VLM's own
    # sentence addressed to the person). Empty string means "none" — the
    # UI falls back to the terse GO!/STOP! banner alone.
    message: str = ""

    @property
    def all_clear(self) -> bool:
        """GO! iff at least one person is present and every person has a vest."""
        return len(self.detections) > 0 and all(d.wearing_vest for d in self.detections)

    @property
    def status(self) -> str:
        if not self.detections:
            return "NO PERSON"
        return "GO" if self.all_clear else "STOP"


class VestDetector:
    """Base class for a pluggable safety-vest detection backend."""

    name: str = "base"

    def warmup(self) -> None:
        """Optional: load weights / run a dummy inference to pay startup cost once."""

    def unload(self) -> None:
        """Optional: free backend resources (GPU memory in particular) when
        going idle for a while. Default no-op — cheap backends (e.g. YOLO's
        small footprint) don't need it. VLM backends should override this:
        idle-pausing a Sentry's inference loop stops new work but does NOT
        by itself free VRAM, and multiple multi-GB VLMs left resident
        simultaneously fills an 8GB card fast enough that even a genuinely
        idle model starves the active one of headroom for its own working
        buffers — slower from memory pressure, not scheduling contention.
        Called by the server after IDLE_TIMEOUT_S with no polls; `warmup()`
        is called again on the next poll to reload."""

    def infer(self, frame_bgr: np.ndarray) -> DetectionResult:
        raise NotImplementedError

    def timed_infer(self, frame_bgr: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        result = self.infer(frame_bgr)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        result.backend = self.name
        return result
