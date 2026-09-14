from __future__ import annotations

from .base import VestDetector

_REGISTRY: dict[str, type[VestDetector]] = {}


def _lazy_register() -> None:
    if _REGISTRY:
        return
    from .vlm_qwen2vl import QwenVL7BDetector, QwenVLDetector
    from .yolo_hsv import YoloHsvDetector

    _REGISTRY["yolo_hsv"] = YoloHsvDetector
    # vlm_qwen2vl is the edge deployment candidate, matching Hailo's own
    # Qwen2-VL-2B-Instruct reference demo for the AI HAT+ 2 / Hailo-10H.
    # vlm_moondream.py still exists (archived, not registered here) as a
    # comparison baseline you can re-enable — see README "Archived:
    # moondream2 comparison baseline".
    _REGISTRY["vlm_qwen2vl"] = QwenVLDetector
    # vlm_qwen2vl_7b is an accuracy-comparison track only (not edge-viable):
    # Qwen2-VL-7B-Instruct, 4-bit quantized. Needs ~16GB+ VRAM headroom
    # (e.g. brannigan's RTX 4080) — registering it is harmless on the 8GB
    # dev laptop since detectors load lazily, but warmup() will raise there.
    _REGISTRY["vlm_qwen2vl_7b"] = QwenVL7BDetector


def list_detectors() -> list[str]:
    _lazy_register()
    return sorted(_REGISTRY)


def get_detector(name: str, **kwargs) -> VestDetector:
    _lazy_register()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown detector '{name}'. Available: {list_detectors()}")
    return _REGISTRY[name](**kwargs)
