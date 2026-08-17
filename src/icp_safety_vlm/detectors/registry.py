from __future__ import annotations

from .base import VestDetector

_REGISTRY: dict[str, type[VestDetector]] = {}


def _lazy_register() -> None:
    if _REGISTRY:
        return
    from .vlm_qwen2vl import QwenVLDetector
    from .yolo_hsv import YoloHsvDetector

    _REGISTRY["yolo_hsv"] = YoloHsvDetector
    # vlm_qwen2vl is the sole active VLM backend: it's the edge deployment
    # candidate, matching Hailo's own Qwen2-VL-2B-Instruct reference demo
    # for the AI HAT+ 2 / Hailo-10H. vlm_moondream.py still exists
    # (archived, not registered here) as a comparison baseline you can
    # re-enable — see README "Archived: moondream2 comparison baseline".
    _REGISTRY["vlm_qwen2vl"] = QwenVLDetector


def list_detectors() -> list[str]:
    _lazy_register()
    return sorted(_REGISTRY)


def get_detector(name: str, **kwargs) -> VestDetector:
    _lazy_register()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown detector '{name}'. Available: {list_detectors()}")
    return _REGISTRY[name](**kwargs)
