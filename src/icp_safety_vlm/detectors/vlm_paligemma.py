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

**Verdict prompts (person/vest counts) carried over from vlm_qwen2vl.py's
count-based structure — partially worked, partially didn't, both found
by direct testing, not assumed.** Single-person verdicts tested correctly
early on (STOP, 398ms, no acquiescence-bias tell). But the original
`_Q_VESTS` compound phrasing **undercounted with 2+ people** — a real bug
found live 2026-09-14 (user report: multi-person frames should only GO
when *everyone* has a vest, which is what `infer()`'s `n_vests >=
n_people` already implements, but the count feeding it was wrong): on a
frame with 2 people both properly wearing vests, it answered `"1"`,
producing a false STOP. Root-caused to two specific fragility triggers —
the color list (worse: broke even the *single*-person case down to "0")
and the word "properly" (broke only the 2-person case) — neither a
"compound questions are fragile in general" problem, both isolated by
testing each clause's removal independently. Current `_Q_VESTS` (below)
was verified correct on all three of: 2-person-both-worn, 1-person-worn,
1-person-held-not-worn. Still needs broader adversarial re-testing (a
non-compliant person mixed with a compliant one, 3+ people) before this
is "verified" rather than "correct on the cases tried so far" — the same
caveat vlm_qwen2vl.py's docstring holds itself to.

**Message generation could NOT be carried over — confirmed empirically,
not just suspected.** The Qwen2-VL-style prompt ("Speaking directly to the
person... explain in one full sentence...") gets a flat refusal from
PaliGemma-mix: *"Sorry, as a base VLM I am not trained to answer this
question."* This isn't a wording problem — it reproduces on multiple
rephrasings of the same compound instruction (tried: a direct restatement,
"describe the safety vest situation", "describe the person and what
they're wearing" — all refused identically). PaliGemma-mix reliably
answers short, single-clause VQA questions (`"caption en"` →
*"A man with glasses on his face looking at the camera."*; "what is the
person wearing on their upper body?" → *"sweatshirt"*) but not open-ended
multi-clause generation instructions — a real architectural difference
from Qwen2-VL's instruct-chat tuning, not a prompt-engineering gap to
close with more iteration.

**Fix**: ask a plain, short VQA question about what's actually on the
person's torso (mirrors the diagnostic vlm_qwen2vl.py's docstring
describes using to isolate Qwen2-VL's acquiescence bias — plain
description, no classification/instruction framing), then compose the
final advisory sentence in *code* around that genuine model-perceived
detail — a hybrid of `vlm_moondream.py`'s fixed-template approach and
vlm_qwen2vl.py's fully-model-authored one. The garment description is
real model output; the sentence wrapper is not. Document this
distinction honestly if asked whether this track's messages are
"model-authored" the same way vlm_qwen2vl's are — they aren't, and
overselling that would misrepresent the demo.
"""

from __future__ import annotations

import re

import numpy as np

from .base import Detection, DetectionResult, VestDetector

_Q_PEOPLE = "answer en How many people are visible in this image?"
# This exact phrasing was chosen empirically after the original compound
# version (below, kept for the record) undercounted with 2+ people — see
# module docstring "Multi-person undercounting" for the full diagnostic.
# Two specific clauses turned out to be the fragility triggers, not
# compound phrasing in general: the color list ("orange, yellow, or lime
# green") broke even the single-person case, and "properly" alone broke
# the 2-person case. This phrasing keeps "high-visibility" (excludes a
# plain suit vest) and the explicit held-vs-worn distinction (verified
# against the adversarial held-not-worn test case), drops both fragility
# triggers, and was verified correct on all three: 2-person-both-worn,
# 1-person-worn, 1-person-held-not-worn.
_Q_VESTS = (
    "answer en How many people are wearing a high-visibility safety vest "
    "on their body, not just holding one?"
)
# Documented negative result, not currently used: the original compound
# phrasing undercounted 2+ people (answered "1" for a frame with 2 people
# both properly wearing vests, confirmed against "how many people are
# wearing a safety vest?" correctly answering "2" on the same frame):
#   "answer en How many people in this image are properly wearing a
#    high-visibility orange, yellow, or lime green safety vest on their
#    body? A vest only held in a hand or hanging nearby does not count."
# Plain, single-clause VQA question — PaliGemma-mix answers this
# reliably (tested: "sweatshirt", "a yellow vest") where a compound
# generation instruction gets refused outright (see module docstring).
# The final advisory sentence is built around this genuine answer in
# _compose_message() below, not generated by the model itself.
_Q_TORSO_DESCRIPTION = "answer en What is the person wearing on their upper body?"

_MESSAGE_STOP_TEMPLATE = (
    "I see you're wearing {description}, not a safety vest - please put "
    "on a proper high-visibility safety vest before proceeding."
)
_MESSAGE_GO_TEMPLATE = (
    "I see you're wearing {description} - that's a proper safety vest, "
    "you may proceed."
)

_VOWEL_START = ("a", "e", "i", "o", "u")


_REDUNDANT_PREFIX = re.compile(
    r"^(the person is|they('re| are)?|he('s| is)?|she('s| is)?)\s+wearing\s+",
    re.IGNORECASE,
)


def _compose_message(description: str, compliant: bool) -> str:
    description = description.strip().rstrip(".") or "something unclear"
    # Usually a bare noun phrase ("hoodie", "a yellow vest"), but strip a
    # redundant "wearing" clause on the rare longer, full-sentence answer
    # so it doesn't double up with this function's own "wearing {...}".
    description = _REDUNDANT_PREFIX.sub("", description).strip() or description
    # PaliGemma's short VQA answers are often a bare noun ("hoodie",
    # "vest") without an article — add one so the composed sentence reads
    # naturally ("wearing a hoodie" vs. "wearing hoodie").
    first_word = description.split(" ", 1)[0].lower()
    needs_article = (
        description[:1].isalpha()
        and not first_word.endswith("s")
        and first_word not in ("a", "an", "the")
    )
    if needs_article:
        article = "an" if first_word.startswith(_VOWEL_START) else "a"
        description = f"{article} {description}"
    # cv2.putText's Hershey fonts are ASCII-only — a non-ASCII dash here
    # renders as "?" glyphs on the annotated video frame (see
    # server/app.py's _draw_wrapped_text), so templates use a plain "-".
    template = _MESSAGE_GO_TEMPLATE if compliant else _MESSAGE_STOP_TEMPLATE
    return template.format(description=description)


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

        description = self._ask(image, _Q_TORSO_DESCRIPTION, max_new_tokens=16)
        message = _compose_message(description, compliant)

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

    **On brannigan specifically, this is a memory-footprint trade only —
    not a speed or accuracy win, confirmed by direct measurement
    (2026-09-14, same frame, back-to-back).** vs. `PaliGemmaDetector`:
    VRAM 2.62GB vs. 6.08GB (the real, intended win), but latency ~318ms
    vs. ~295ms — quantized is *slightly slower*, not faster, and output
    (verdict + message) was identical on every test frame, which makes
    sense since it's the same weights at lower precision, not a different
    or "better" model. The latency regression has a concrete cause, not
    generic quantization overhead: bitsandbytes warns
    `inner dimension (4304) is not aligned for fast kernel with
    blocksize=64, falling back to slower implementation` for this model's
    layer dimensions — the dequantize-then-matmul fallback path costs more
    than the memory-bandwidth savings recover, and brannigan's 16GB has no
    memory pressure for quantization to relieve in the first place. This
    finding is brannigan/bitsandbytes-CUDA-specific and may not transfer
    to the Jetson's ARM/Ampere bitsandbytes build or its actually-
    constrained memory budget — re-measure there rather than assume either
    outcome.

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


class PaliGemma10BDetector(PaliGemmaDetector):
    """Same prompts/pipeline as PaliGemmaDetector, scaled up to
    PaliGemma 2 **10B**-mix — a same-family larger-model comparison point,
    to see what a bigger model buys (or doesn't) over the 3B default, now
    that brannigan's 16GB has headroom to try.

    **Must run quantized to fit at all on brannigan** — 10B params at
    bf16 is ~20GB, which exceeds the RTX 4080's 16GB outright, so unlike
    `PaliGemma2QuantizedDetector` (where quantization is optional on this
    card and only pays off on the Jetson), here it's load-bearing on
    brannigan too. Uses **8-bit**, not 4-bit: `PaliGemma2QuantizedDetector`'s
    docstring documents a measured 4-bit slowdown from a bitsandbytes
    kernel-alignment fallback on this model family, and 8-bit LLM.int8()
    fits comfortably in 16GB (~10-11GB expected) without needing to drop
    to 4-bit's more aggressive (and, on this hardware, slower) path — keeps
    the 3B-vs-10B comparison from being confounded by a second, unrelated
    quantization-mode variable. Not yet measured whether 8-bit hits a
    similar kernel-alignment issue; check the load-time bitsandbytes
    warnings the way `PaliGemma2QuantizedDetector`'s finding was found.

    Comparison is therefore "3B at full bf16" vs. "10B at 8-bit" — a real
    limitation of this being an accuracy/capability comparison, not a
    clean same-precision one; call this out explicitly if reporting
    numbers from it, don't present it as controlling for precision.

    **Measured result (2026-09-14, live + direct diagnostic on the same
    worn-vest frame): 10B is worse on every axis for this task, not
    better.** 8-bit loaded clean (no bitsandbytes kernel-alignment
    warning, unlike the 4-bit 3B variant), but:
    - VRAM: 10.75GB vs. 3B's 6.08GB.
    - Latency: ~1.18s/frame steady-state vs. 3B's ~295ms — **~4x slower**.
    - Accuracy on the exact question this architecture depends on: **wrong**.
      10B correctly answers direct questions about the same frame ("yes"
      to wearing a safety vest, "yellow" for its color) but answers `"0"`
      to both the exact `_Q_VESTS` compound prompt AND a simplified
      single-clause version ("How many people are wearing a yellow safety
      vest?") on a frame with one person clearly, properly wearing a
      yellow vest — a literal digit this time, not the 3B word-vs-digit
      parsing bug (`_parse_int` handles both correctly; this is the model
      giving a wrong count, not an unparseable one). It also refuses
      `"caption en"` outright, where 3B answers it fine — a real
      behavioral difference between the two mix checkpoints' tuning, not
      a fixed scale-up of the same behavior.

    Not pursued further per user decision (2026-09-14): keep this as a
    documented negative result rather than sink time into redesigning a
    10B-specific counting prompt. If revisited later, the counting-prompt
    fragility observed across three different models now (Qwen2-VL's
    original acquiescence bias, 3B's word-form answers, 10B's flat wrong
    count) suggests the fix is unlikely to be a wording tweak — worth
    trying a structurally different verdict signal (e.g., per-person
    yes/no instead of a count) before assuming more prompt iteration will
    converge."""

    name = "vlm_paligemma_10b"

    def __init__(self, model_id: str = "google/paligemma2-10b-mix-224", device: str | None = None):
        super().__init__(model_id=model_id, device=device)

    def warmup(self) -> None:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, PaliGemmaForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{self.name} requires a CUDA GPU for 8-bit inference (none detected)"
            )

        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        self._model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id, quantization_config=quant_config, device_map="cuda"
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._device = "cuda"


_WORD_NUMBERS = {
    "zero": 0, "no": 0, "none": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _parse_int(answer: str) -> int:
    """Best-effort parse of a numeric answer; falls back to 0 rather than raising
    if the model returns something unexpected.

    **Real bug this fixes, found live (2026-09-14)**: PaliGemma answered
    the vest-count question with the word "one" rather than the numeral
    "1" on a frame with a person clearly wearing a yellow AS/NZS-style
    hi-vis vest. The original digit-only regex silently parsed that as 0
    ("no vest"), so a correctly-perceiving model produced a wrong STOP
    verdict purely from an answer-format mismatch — not a
    perception/prompt-design failure the way the message-generation
    refusal was. Confirmed by direct testing: simpler questions ("Is the
    person wearing a safety vest?", "What color is the vest?") all
    answered correctly ("yes", "yellow"); only the exact multi-clause
    _Q_VESTS phrasing elicited a word instead of a digit. Handling both
    forms here is more robust than trying to prompt-engineer PaliGemma
    into never answering in words, since nothing observed so far
    predicts which form a given question will get."""
    match = re.search(r"\d+", answer)
    if match:
        return int(match.group(0))
    first_word = answer.strip().lower().split(" ", 1)[0].strip(".,!?")
    return _WORD_NUMBERS.get(first_word, 0)
