"""Track B — VLM: Qwen2-VL-2B-Instruct.

This is the model to treat as the real edge deployment candidate: Hailo's
own reference app for the AI HAT+ 2 / Hailo-10H (`vlm_chat.py` in Hailo
Apps) runs Qwen2-VL-2B-Instruct on exactly this Pi 5 + Camera Module 3
stack. Developing against the same architecture family here removes "will
this even compile for Hailo" as a variable later — see README for the
Hailo-10H specifics and the independent benchmark caveats (decode
throughput on-chip isn't guaranteed to beat CPU for small models; the real
lever is minimizing generated tokens, not raw TOPS).

Three calls per frame when a person is present (one when not):

1. **Person count** — terse numeric prompt, `max_new_tokens` capped tiny
   (8). Decides NO PERSON without needing the model to commit to a vest
   verdict on an empty scene.
2. **Vest count** — also terse and numeric ("how many are *properly*
   wearing one"). This decides GO/STOP.
3. **Message** — only asked once the verdict from (2) is known, and the
   prompt is told which verdict to phrase, not asked to judge compliance
   itself. Real free-text generation (not a fixed template) — but grounded,
   not self-classifying.

That last split (count-based verdict, model only phrases it) isn't just
tidiness — it's the fix for a real accuracy bug. An earlier version
combined classification and message into a single call where the model had
to open its own answer with the literal word GO or STOP. Tried three
prompt framings for that (a direct "answer GO or STOP", a
WORN/HELD/NONE multiple-choice, and a "respond in this style: example"
version) — **all three were unreliable, and each failed a different way**:
a plain yes/no question defaulted to "yes" regardless of the image; the
multiple-choice answered inconsistently across repeated calls on the exact
same frame; the styled-example version reliably produced a full sentence
but sometimes got the actual GO/STOP wrong on frames where the person was
demonstrably not wearing a vest (verified against the raw captured frame,
not a staleness artifact). This is acquiescence bias — well documented in
small VLMs/LLMs, where a direct classification-style question gets
answered from the *shape* of the question more than the image. The tell:
asking the same model to plainly *describe* what's on the person's torso
(no classification framing) got it right every time — the model can see
correctly, it just can't be trusted to self-classify in one shot. Asking
for **counts** turned out to dodge the bias just as reliably as asking for
a description, and counts are trivial to parse and to build a GO/STOP
banner from — so that's the architecture: count-based ground truth, and a
*separate*, *grounded* call for the sentence, mirroring the earlier lesson
that message generation should never re-judge what code already knows.
"""

from __future__ import annotations

import numpy as np

from .base import Detection, DetectionResult, VestDetector

_Q_PEOPLE = "How many people are visible in this image? Answer with just a number."
_Q_VESTS = (
    "How many people in this image are properly wearing a high-visibility "
    "orange, yellow, or lime green safety vest on their body? A vest only "
    "held in a hand or hanging nearby does not count. Answer with just a "
    "number."
)
_Q_MESSAGE_STOP = (
    "Speaking directly to the person and only about their safety vest, "
    "explain in one full sentence that they are not wearing one and must "
    "put on a proper safety vest before proceeding."
)
_Q_MESSAGE_GO = (
    "Speaking directly to the person and only about their safety vest, "
    "explain in one full sentence that they are properly wearing their "
    "safety vest and may proceed."
)


class QwenVLDetector(VestDetector):
    name = "vlm_qwen2vl"

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct", device: str | None = None):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def warmup(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=[image], padding=True, return_tensors="pt").to(
            self._device
        )
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

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


def _parse_int(answer: str) -> int:
    """Best-effort parse of a numeric answer; falls back to 0 rather than raising
    if the model returns something unexpected."""
    import re

    match = re.search(r"\d+", answer)
    return int(match.group(0)) if match else 0
