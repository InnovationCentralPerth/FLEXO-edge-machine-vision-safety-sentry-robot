"""Track A — "old school": YOLOv8 person detector + HSV hi-vis color heuristic.

COCO (what stock YOLOv8 is trained on) has no "safety vest" class, so the
classic pipeline is: detect the person box, crop the torso region, and check
whether enough pixels fall in the hi-vis orange/yellow (or lime-green) HSV
range. This mirrors how a lot of shipped PPE-compliance products actually
work when they don't have a custom-labeled vest dataset, and it's a fair,
fast, CPU/GPU-cheap baseline to compare a VLM against.
"""

from __future__ import annotations

import numpy as np

from .base import Detection, DetectionResult, VestDetector

# HSV ranges (OpenCV: H 0-179, S/V 0-255) for common hi-vis vest colors.
_HSV_RANGES = [
    # hi-vis orange
    ((5, 120, 120), (18, 255, 255)),
    # hi-vis yellow
    ((22, 100, 120), (35, 255, 255)),
    # hi-vis lime/green
    ((36, 80, 120), (70, 255, 255)),
]

_MIN_VEST_PIXEL_FRACTION = 0.12  # of torso-crop pixels that must match hi-vis color


def _torso_crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    h = y2 - y1
    # torso ~ upper 60% of the person box, vertically, where a vest sits
    ty1 = y1 + int(0.10 * h)
    ty2 = y1 + int(0.70 * h)
    return frame[max(ty1, 0) : max(ty2, 0), x1:x2]


def _hivis_fraction(crop_bgr: np.ndarray) -> float:
    import cv2

    if crop_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _HSV_RANGES:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return float(np.count_nonzero(mask)) / float(mask.size)


class YoloHsvDetector(VestDetector):
    name = "yolo_hsv"

    def __init__(self, model_name: str = "models/yolov8n.pt", conf: float = 0.4, device: str | None = None):
        self.model_name = model_name
        self.conf = conf
        self.device = device
        self._model = None

    def warmup(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO(self.model_name)
        # one dummy inference to trigger CUDA context / graph build
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self._model.predict(dummy, verbose=False, device=self.device)

    def infer(self, frame_bgr: np.ndarray) -> DetectionResult:
        if self._model is None:
            self.warmup()
        assert self._model is not None

        results = self._model.predict(
            frame_bgr, conf=self.conf, classes=[0], verbose=False, device=self.device
        )
        detections: list[Detection] = []
        if results:
            r = results[0]
            for box in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                score = float(box.conf[0])
                crop = _torso_crop(frame_bgr, (x1, y1, x2, y2))
                frac = _hivis_fraction(crop)
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        wearing_vest=frac >= _MIN_VEST_PIXEL_FRACTION,
                        confidence=score,
                    )
                )
        return DetectionResult(detections=detections)
