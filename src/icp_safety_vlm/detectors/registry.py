from __future__ import annotations

from .base import VestDetector

_REGISTRY: dict[str, type[VestDetector]] = {}


def _lazy_register() -> None:
    if _REGISTRY:
        return
    from .vlm_paligemma import PaliGemma2QuantizedDetector, PaliGemma10BDetector, PaliGemmaDetector
    from .yolo_hsv import YoloHsvDetector

    _REGISTRY["yolo_hsv"] = YoloHsvDetector
    # vlm_paligemma is the active VLM track (On-Prem, full precision) —
    # PaliGemma 2 3B, replacing Qwen2-VL (Alibaba-origin) for a non-CN-origin
    # open model. See system-design-integration.md "Model choice" for the
    # rationale, and vlm_paligemma.py's module docstring for why its
    # prompt design (inherited from vlm_qwen2vl.py's count-based structure)
    # is not yet independently verified against acquiescence bias.
    _REGISTRY["vlm_paligemma"] = PaliGemmaDetector
    # vlm_paligemma_quantized: same base checkpoint, 4-bit quantized — the
    # On-Device (Jetson Orin Nano Super) candidate. Needs a CUDA GPU;
    # registering it is harmless on a CPU-only box since detectors load
    # lazily, but warmup() will raise there. See system-design-
    # integration.md "Solution B" for the Jetson's tighter (shared,
    # unified) 7.4GB memory budget that makes quantization load-bearing
    # rather than just a speed optimization there.
    _REGISTRY["vlm_paligemma_quantized"] = PaliGemma2QuantizedDetector
    # vlm_paligemma_10b: same family, scaled up to 10B (8-bit quantized —
    # required to fit brannigan's 16GB at all, not optional). Larger-model
    # comparison point, not an edge candidate; see vlm_paligemma.py's
    # PaliGemma10BDetector docstring for why this isn't a same-precision
    # comparison against vlm_paligemma (3B fp16).
    _REGISTRY["vlm_paligemma_10b"] = PaliGemma10BDetector
    # Archived: vlm_qwen2vl.py (QwenVLDetector / QwenVL7BDetector) and
    # vlm_moondream.py still exist but aren't registered here — kept as
    # comparison baselines and for their documented prompt-design lessons
    # (see README "Prompt design (VLM detection)"), not because either is
    # broken. Re-enable either with one line here if you need the
    # comparison back.


def list_detectors() -> list[str]:
    _lazy_register()
    return sorted(_REGISTRY)


def get_detector(name: str, **kwargs) -> VestDetector:
    _lazy_register()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown detector '{name}'. Available: {list_detectors()}")
    return _REGISTRY[name](**kwargs)
