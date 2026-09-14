"""Track B — VLM: PaliGemma 2 (google/paligemma2-3b-mix-224).

Replaces Qwen2-VL-2B-Instruct as the active VLM track (see
system-design-integration.md's "Model choice" section for the full
rationale) — Qwen2-VL is Alibaba-origin; this phase of the project
requires a non-CN-origin open model. PaliGemma 2 is Google/DeepMind-origin,
Apache-style open weights (Gemma license — must be accepted on Hugging
Face once per account before the first download, or `warmup()` fails with
a 403 rather than a clear license-gate message).

Same base checkpoint is meant to run on both solutions in
system-design-integration.md: this module at full precision for the
On-Prem (RTX 4080) track, and a quantized copy of the same 3B checkpoint
for the On-Device (Jetson Orin Nano Super) track — see
`PaliGemma2QuantizedDetector` below, mirroring how `vlm_qwen2vl.py`'s
`QwenVL7BDetector` subclassed `QwenVLDetector` for a different precision.

**Architecturally different from Qwen2-VL in a way that matters for
prompting**: PaliGemma is a single-turn image + text-prefix -> text
completion model, not a chat model — no `apply_chat_template`, no
"assistant" framing to fight against. The "mix" checkpoint (vs. "pt")
is instruction-tuned across a mixture of tasks including open VQA, so it
accepts a natural-language question with an `"answer en "` prefix rather
than requiring pretraining-specific task strings.

**Prompt design is NOT yet re-verified against PaliGemma.** vlm_qwen2vl.py
documents a real, three-iteration acquiescence-bias fix (self-classifying
questions get answered from the shape of the question, not the image;
count-based questions dodge it). This module reuses that same count-based
structure as a starting hypothesis, not a verified transfer — re-run the
same kind of adversarial testing (known-STOP frames, repeated calls on the
same frame) before trusting it in a demo. See system-design-integration.md
"Operational gaps" item 5.
"""

from __future__ import annotations

import numpy as np

from .base import Detection, DetectionResult, VestDetector

_Q_PEOPLE = "answer en How many people are visible in this image?"
_Q_VESTS = (
    "answer en How many people in this image are properly wearing a "
    "high-visibility orange, yellow, or lime green safety vest on their "
    "body? A vest only held in a hand or hanging nearby does not count."
)
_Q_MESSAGE_STOP = (
    "answer en Speaking directly to the person and only about their "
    "safety vest, explain in one full sentence that they are not "
    "wearing one and must put on a proper safety vest before proceeding."
)
_Q_MESSAGE_GO = (
    "answer en Speaking directly to the person and only about their "
    "safety vest, explain in one full sentence that they are properly "
    "wearing their safety vest and may proceed."
)


class PaliGemmaDetector(VestDetector):
    name = "vlm_paligemma"

    def __init__(self, model_id: str = "google/paligemma2-3b-mix-224", device: str | None = None):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def warmup(self) -> None:
        import torch
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self._model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._device = device

    def unload(self) -> None:
        import gc

        import torch

        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ask(self, image, prompt: str, max_new_tokens: int) -> str:
        import torch

        assert self._model is not None and self._processor is not None

        inputs = self._processor(text=prompt, images=image, return_tensors="pt").to(
            self._device, self._model.dtype
        )
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        # PaliGemma's generate() output includes the input prompt tokens
        # (unlike Qwen2-VL's chat-template path) — slice them off before
        # decoding, or the "answer" comes back prefixed with the question.
        new_tokens = generated_ids[0][input_len:]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()

    def infer(self, frame_bgr: np.ndarray) -> DetectionResult:
        if self._model is None:
            self.warmup()

        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        n_people = _parse_int(self._ask(image, _Q_PEOPLE, max_new_tokens=8))
        if n_people == 0:
            return DetectionResult(detections=[], message="")

        n_vests = min(
            _parse_int(self._ask(image, _Q_VESTS, max_new_tokens=8)),
            n_people,
        )
        compliant = n_vests >= n_people

        message_q = _Q_MESSAGE_GO if compliant else _Q_MESSAGE_STOP
        message = self._ask(image, message_q, max_new_tokens=40).strip()

        h, w = frame_bgr.shape[:2]
        detections = [
            Detection(bbox=(0, 0, w, h), wearing_vest=compliant, confidence=1.0)
            for _ in range(n_people)
        ]
        return DetectionResult(detections=detections, message=message)


class PaliGemma2QuantizedDetector(PaliGemmaDetector):
    """Same base checkpoint and prompts as PaliGemmaDetector, 4-bit
    quantized — the On-Device (Jetson Orin Nano Super) candidate per
    system-design-integration.md. The Jetson's 7.4GB is *unified*
    CPU/GPU memory (no separate VRAM pool, unlike brannigan's 16GB VRAM),
    so it has to share that budget with the OS, camera capture, and the
    server process — quantization here is load-bearing for fitting at
    all, not just a speed optimization.

    Requires a CUDA device and the `vlm` extra's `bitsandbytes` dependency
    — see system-design-integration.md's "Operational gaps" item 4 for the
    still-open question of whether bitsandbytes-on-transformers or a
    Jetson-native TensorRT export is the right long-term path; this class
    is the bitsandbytes prototype, chosen first because it reuses the
    On-Prem code path with minimal new work."""

    name = "vlm_paligemma_quantized"

    def warmup(self) -> None:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, PaliGemmaForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{self.name} requires a CUDA GPU for 4-bit inference (none detected)"
            )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id, quantization_config=quant_config, device_map="cuda"
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._device = "cuda"


def _parse_int(answer: str) -> int:
    """Best-effort parse of a numeric answer; falls back to 0 rather than raising
    if the model returns something unexpected."""
    import re

    match = re.search(r"\d+", answer)
    return int(match.group(0)) if match else 0
