# ICP Safety VLM Sentry

Prototype safety-vest sentry: a browser page shows a big **GO!** / **STOP!**
banner based on whether every person in a live camera feed is wearing a
high-visibility safety vest, plus (VLM track) a one-sentence, on-topic
message written by the model itself.

Built to compare detection strategies head-to-head on the same hardware,
before committing to one for the edge target.

> **Current status (2026-09-14)**: this project is now two solutions —
> On-Prem (this doc, running on brannigan's RTX 4080) and On-Device
> (a standalone Jetson Orin Nano Super) — see
> **[system-design-integration.md](system-design-integration.md)** for the
> integrated architecture. The active VLM track is now **PaliGemma 2**
> (`detectors/vlm_paligemma.py`), replacing Qwen2-VL for non-CN-origin
> model provenance; Qwen2-VL and moondream2 are both archived (code still
> present, not registered by default). Most of this README predates that
> split and describes the original single-machine prototype phase — the
> Hailo-10H research and the Qwen2-VL prompt-design writeup below are
> still accurate *as history* and worth reading for the lessons, but
> `vlm_qwen2vl` is no longer the backend you get by default. See
> `vlm_paligemma.py`'s module docstring for the current model's own
> (differently-shaped) prompt-design notes.

## On-Prem (brannigan) results log — PaliGemma 2

Single place for "what model, what prompts, what we found" on the current
On-Prem track. Full detail/code lives in `detectors/vlm_paligemma.py`'s
docstrings and `system-design-integration.md`'s benchmark table — this is
the summary.

**Model**: `google/paligemma2-3b-mix-224` (`vlm_paligemma`, fp16) is the
active default. Two comparison variants also registered:
`vlm_paligemma_quantized` (same 3B, 4-bit) and `vlm_paligemma_10b`
(`google/paligemma2-10b-mix-224`, 8-bit — required just to fit brannigan's
16GB, not optional).

**Prompts** (`vlm_paligemma.py`): person-count and vest-count are terse
VQA questions, decided first; a torso-description VQA answer is composed
into the final advisory sentence in code (not model-generated free text —
PaliGemma-mix refuses open-ended generation instructions, see below).
The vest-count prompt went through one real fix:
- *Before* (undercounted with 2+ people): "How many people in this image
  are properly wearing a high-visibility orange, yellow, or lime green
  safety vest on their body? A vest only held in a hand or hanging nearby
  does not count."
- *After* (verified correct on 2-person-both-worn, 1-person-worn, and
  1-person-held-not-worn): "How many people are wearing a high-visibility
  safety vest on their body, not just holding one?"
- Root cause, isolated by testing each clause's removal independently:
  the color list broke even the single-person case; the word "properly"
  alone broke the 2-person case. Not a general "shorten it" fix — a
  specific two-clause diagnosis.

**Findings, in the order found**:
1. Message generation cannot reuse Qwen2-VL's approach — PaliGemma-mix
   flatly refuses compound generation instructions ("explain in one full
   sentence..."). Fixed with a short VQA description + code-composed
   sentence template.
2. `_parse_int()` silently dropped word-form counts ("one" parsed as 0,
   not 1) — caused a real false STOP on a correctly-worn vest. Fixed to
   handle both digit and word forms.
3. Stability pass (single person, 37/37 correct): 15/15 `GO` while vest
   worn (yellow, moving), 8/8 `STOP` vest off, 6/6 `STOP` vest held-not-worn
   (the adversarial case), plus a separate confirmed `GO` on the orange
   vest.
4. 4-bit quantization (`vlm_paligemma_quantized`) is a **memory trade
   only, not a speed win**, on brannigan: 2.62GB vs. 6.08GB VRAM, but
   ~318ms vs. ~295ms latency (slightly *slower* — bitsandbytes falls back
   to a slower kernel for this model's layer dimensions). May not predict
   the Jetson result (different bitsandbytes build, genuinely
   memory-constrained there).
5. The 10B same-family model (`vlm_paligemma_10b`) is **worse on every
   measured axis**: more VRAM (10.75GB), ~4x slower (~1.18s/frame), and
   wrong on the vest-count question (answers "0" on a clearly, properly
   worn vest) despite correctly answering simpler direct questions about
   the same frame. Kept as a documented negative result, not pursued
   further.
6. Multi-person undercounting (user-reported): the original vest-count
   prompt answered "1" for a frame with 2 people both properly wearing
   vests, producing a false STOP. Fixed per the prompt change above;
   re-verified through the actual deployed detector code (not just the
   raw prompt) on saved test frames: 2-person-both-worn → GO,
   1-person-worn → GO, 1-person-held-not-worn → STOP. **Not yet
   re-verified live with two people since the fix deployed** — the saved-frame
   re-test is real but a live re-check is still worth doing when two
   people are available again.

## Hardware

- **Dev laptop:** Ubuntu 24 / WSL2, RTX 3000 Ada 8GB VRAM (driver reports
  CUDA 13.2, toolkit 12.0). Has the webcam; runs `scripts/camera_relay.py`
  when the server itself runs on brannigan (see [Distributed
  setup](#distributed-setup-brannigan-rtx-4080) below).
- **brannigan (RTX 4080, 16GB VRAM):** where the server + dashboard +
  inference run day-to-day now — enough VRAM headroom for the
  `vlm_qwen2vl_7b` accuracy-comparison track, which doesn't fit on the
  laptop's 8GB card. No camera attached; sources frames from the laptop's
  relay over Tailscale.
- **Final edge target:** Raspberry Pi 5 + **AI HAT+ 2 (Hailo-10H)** + Pi
  Camera Module 3, standalone, no laptop/GPU box attached. See
  [Edge target: Hailo-10H](#edge-target-hailo-10h) below — it's a
  purpose-built on-device GenAI accelerator, not a classic CNN-only NPU,
  which materially shapes the optimization plan.

## Distributed setup (brannigan RTX 4080)

The server, dashboard, and both VLM tracks now run on **brannigan**
(`stanley@brannigan.taila277ca.ts.net`, Tailscale), which has no camera of
its own. This laptop's webcam is relayed to it over the tailnet instead of
attaching hardware directly.

**One-time setup on brannigan:**
```bash
ssh stanley@brannigan.taila277ca.ts.net
cd ~/projects/icp_safety_vlm   # already `git clone`d from this repo's origin
uv sync --extra yolo --extra vlm
```
brannigan's NVIDIA driver needed a reboot to clear a driver/library version
mismatch (`nvidia-smi` failing) as of this port — confirm `nvidia-smi` works
before expecting CUDA to be picked up; `torch.cuda.is_available()` silently
falls back to CPU otherwise.

**Every run, two processes:**
```bash
# 1. On the laptop (has the webcam) — see WSL2 webcam passthrough above
#    if /dev/video* isn't showing up first:
uv run python scripts/camera_relay.py

# 2. On brannigan — point it at the laptop's relay instead of a local device:
CAMERA_SOURCE=http://<laptop-tailnet-name>.taila277ca.ts.net:8100/frame \
    uv run uvicorn icp_safety_vlm.server.app:app --host 0.0.0.0 --port 8000
```
Then open `http://brannigan.taila277ca.ts.net:8000/` from any machine on the
tailnet. `CAMERA_SOURCE` overrides `CAMERA_INDEX` entirely — see
`server/app.py`'s `Camera` class. Frame latency over Tailscale adds on the
order of tens of ms; irrelevant next to VLM inference time, and still far
below the classic-CV track's budget for anything but a tight real-time
demo.

To go back to running everything locally on one machine (e.g. back on the
dev laptop), just don't set `CAMERA_SOURCE` — the server falls back to
opening `CAMERA_INDEX` directly, as before.

## Two active tracks

- **Track A — classic CV/YOLO** (`detectors/yolo_hsv.py`): YOLOv8n person
  detector + HSV color-threshold heuristic on the torso crop to decide
  hi-vis vest present/absent. COCO has no "vest" class, so this hybrid is
  the standard classic-CV approach — cheap, explainable, no vest-labeled
  training data required. **~10–30ms/frame** on the RTX 3000. Likely the
  safest bet for real-time latency on the edge target regardless of how
  the VLM track pans out — see [Roadmap](#roadmap).
- **Track B — VLM** (`detectors/vlm_qwen2vl.py`): Qwen2-VL-2B-Instruct.
  This is the deployment candidate — Hailo's own reference app for the
  AI HAT+ 2 (`vlm_chat.py` in Hailo Apps) runs this exact model on this
  exact Pi 5 + Camera Module 3 stack, so developing against it now removes
  "will this even compile for Hailo" as a later variable.
  **~1.7–2.9s/frame** (person present; ~0.1–0.3s when no person). Three
  model calls per frame: two reliable counting questions decide the
  verdict, and the model only *phrases* a genuine, free-text sentence for
  a verdict it's already been given — see
  [Prompt design](#prompt-design-vlm-detection) for why the more obvious
  "ask the model to self-classify" design had to be abandoned.
- **Track B (accuracy variant) — `vlm_qwen2vl_7b`** (`detectors/vlm_qwen2vl.py`,
  `QwenVL7BDetector`): same three-call pipeline and prompts as
  `vlm_qwen2vl`, swapped to Qwen2-VL-**7B**-Instruct, 4-bit quantized
  (bitsandbytes NF4) to fit brannigan's 16GB card alongside headroom for a
  second resident backend. **Not an edge deployment candidate** — it's a
  ceiling-accuracy comparison point now that 16GB+ VRAM is available;
  Qwen2-VL-2B stays the Hailo target. Needs a CUDA GPU with real headroom;
  `warmup()` raises outright on the 8GB dev laptop rather than silently
  falling back to a CPU/fp32 crawl.

Both backends implement the same `VestDetector` interface
(`detectors/base.py`), so the server, benchmark script, and future backends
(e.g. a fine-tuned edge model) all plug in the same way.

### Archived: moondream2 comparison baseline

`detectors/vlm_moondream.py` still exists but isn't registered by default.
It was the original VLM track, kept specifically as a comparison point
against Qwen2-VL — but it isn't on Hailo's validated model list for the
AI HAT+ 2, and running both VLMs live at once on an 8GB card leaves so
little VRAM headroom that even a genuinely idle model slows the active one
down from memory pressure (see [Server architecture](#server-architecture)).
Rather than fight that permanently, the default is Qwen2-VL only.

To spin moondream2 (or `vlm_qwen2vl`/`vlm_qwen2vl_7b`, also archived as of
the PaliGemma switch — see the status note at the top of this file) back
up for a benchmark comparison: add it back in `detectors/registry.py`'s
`_lazy_register()` (one line each, still working `VestDetector`s).
**Correction to an earlier version of this note**: `scripts/benchmark.py`
always resolves backends via `get_detector()` (the registry), not a direct
import — so an archived backend needs that one-line re-registration before
`--backend vlm_moondream` (or any other archived name) will work, it does
not run "standalone" without one.

## Setup

```bash
# base deps (server only)
uv sync

# + Track A (YOLO)
uv sync --extra yolo

# + Track B (VLM) — pulls torch/transformers, picks up CUDA automatically
uv sync --extra vlm

# both
uv sync --extra yolo --extra vlm
```

**Note on `transformers` version:** the `vlm` extra pins
`transformers>=4.45,<4.46`. This version happens to support both the
active `PaliGemmaForConditionalGeneration` and the archived
`Qwen2VLForConditionalGeneration`/moondream2 classes; the upper bound is a
holdover from keeping moondream2 loadable in the same environment (its
`trust_remote_code` model class breaks on the `>=5.x` `PretrainedConfig`
internals refactor) — safe to relax if you've dropped that comparison for
good.

**PaliGemma 2 requires accepting Google's Gemma license once per Hugging
Face account** before the first download — visit
https://huggingface.co/google/paligemma2-3b-mix-224 while logged in, then
`hf auth login` (or set `HF_TOKEN`) wherever the server runs. Skipping this
fails `warmup()` with a 403 gated-repo error, not a clear license prompt.
It's a ~6GB download on first use after that, cached by `huggingface_hub`.

## Run

```bash
uv run uvicorn icp_safety_vlm.server.app:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/` (or `?backend=vlm_paligemma` /
`?backend=yolo_hsv` in the URL, or the dropdown on the page). See
[system-design-integration.md](system-design-integration.md) for how this
looks when the server runs on brannigan instead, with the camera relayed
from a laptop over `CAMERA_SOURCE`.

Env vars: `CAMERA_INDEX`, `FRAME_WIDTH`, `FRAME_HEIGHT`, `DETECTOR_BACKEND`.

### WSL2 webcam passthrough

The webcam isn't visible in WSL2 by default. From an elevated **Windows**
PowerShell (not WSL2):

```powershell
winget install usbipd
usbipd list                # find your webcam's BUSID
usbipd bind --busid <ID>
usbipd attach --wsl --busid <ID>
```

If attach fails with `Device busy (exported)`, close whatever's holding the
camera on Windows first (Camera app, Teams/Zoom, Windows Hello, or check
Settings → Privacy → Camera → background app access) — `--force` works but
Windows apps lose the camera while it's bound to WSL2.

Then `ls /dev/video*` in WSL2 should show device nodes. With **mirrored**
WSL2 networking mode, `http://localhost:8000` from the Windows browser
reaches the WSL2 server directly — no port forwarding needed.

**Raw YUYV corruption:** usbipd's virtual USB bus doesn't reliably carry the
bandwidth for raw uncompressed YUYV capture — symptom is large frozen
solid-color blocks in otherwise-live frames. Fixed by forcing MJPEG
(compressed) capture (`cv2.CAP_PROP_FOURCC` = `MJPG`) in `Camera.start()`
in `server/app.py`. If you see corrupted frames on different hardware,
check this first.

## Server architecture

- **One shared `Camera`** owns the single `cv2.VideoCapture` handle and
  continuously reads frames into a shared buffer. V4L2/UVC webcams
  generally only allow one open handle — early versions gave each backend
  its own `VideoCapture`, which meant switching the dropdown starved
  whichever `Sentry` lost the race ("can't open camera by index").
- **One `Sentry` per backend**, created lazily the first time the dropdown
  selects it and kept alive after that. Each pulls frames from `Camera` and
  runs `detector.timed_infer()` in its own thread.
- **Idle backends pause their loop *and* free their memory.** Two separate
  problems showed up running multiple VLM backends live at once (from the
  period before moondream2 was archived — kept here since the fixes are
  general and still apply if you re-enable it, or add a third backend
  later):
  1. *Scheduling contention*: every live `Sentry` used to run its inference
     loop flat-out forever, even for backends nobody was viewing. Three
     backends warm at once, all fighting for the same GPU, turned a
     ~550ms/frame model into ~11s/frame purely from contention.
  2. *Memory pressure*: pausing the loop alone doesn't free VRAM — a
     multi-GB VLM stays resident the whole time it's "idle". Two VLMs'
     worth of weights on an 8GB card left so little headroom for either
     model's own working buffers that even the genuinely-idle-loop backend
     slowed the active one down further, independent of (1).

  Fixed with both an idle-pause *and* an unload: a `Sentry` stops running
  inference and calls `detector.unload()` (frees the model, empties the
  CUDA cache) `IDLE_TIMEOUT_S` (8s) after its last `/status` or `/frame`
  poll, and reloads via `warmup()` the moment it's polled again — costing
  a few seconds, the same as first-time startup. Since the page only polls
  the currently-selected backend, switching the dropdown naturally idles
  the others out. **Implication for benchmarking**: the live server is for
  demoing/eyeballing, not clean latency numbers — use
  `scripts/benchmark.py`, which loads one detector in isolation, for
  anything you're going to report or plot.
- **`GET /frame?backend=...`** returns a single JPEG; the page polls it
  (cache-busted) alongside `/status` every 300ms. Deliberately *not* MJPEG
  (`multipart/x-mixed-replace`) streaming — switching `<img src>` between
  two live multipart streams is flaky across browsers, especially when the
  new stream is slow to send its first byte (VLM warmup). A slow backend
  gains nothing from true 30fps streaming anyway. (`/stream` still exists
  if you want to test raw MJPEG directly.)
- **`GET /status?backend=...`** returns `{status, backend, latency_ms,
  n_people, message, age_ms}` (or `status: "ERROR"` with the exception text
  in `message` if a backend's `warmup()`/inference crashes — surfaced in
  the UI instead of a raw 500). `age_ms` is how stale the current verdict
  is (time since the analyzed frame was captured) — the UI turns this
  orange past 2.5s so a slow VLM's "burst update" behavior is visible
  instead of a STOP banner silently sitting next to a scene that's already
  changed.
- **`GET /debug`** returns each live Sentry's state
  (`{backend, idle_for_s, is_idle, loaded, thread_alive}`) — useful for
  confirming which backend(s) are actually resident/contending for the GPU
  at any given moment.

## Message design

`yolo_hsv` never produces a message (`message: ""` always) — the banner
alone is its output. `vlm_qwen2vl` produces a genuine, model-authored
one-sentence message every frame (e.g. *"The person in the image is not
wearing a safety vest. It is important to wear a proper safety vest
before proceeding..."*), scoped to the vest only. It is **not** a fixed
template — an earlier version used one (`detectors/messages.py` still
holds the two template strings as a fallback/reference, and `messages.py`
is still what `vlm_moondream.py`, the archived backend, uses), but the
brief asked specifically for the VLM's own language, so this was reworked
— see [Prompt design](#prompt-design-vlm-detection) for how, since the
straightforward version of "genuine free text" turned out to be actively
unsafe.

## Prompt design (VLM detection)

Three calls per frame in `vlm_qwen2vl.py` when a person is present (one
when not — see the module docstring for the full reasoning, this is the
short version):

1. **Person count** — terse numeric prompt. Decides NO PERSON.
2. **Vest count** — also terse and numeric ("how many are *properly*
   wearing one, not just holding it"). Decides GO/STOP.
3. **Message** — asked last, and only after (2) already knows the
   verdict. The prompt tells the model which case to phrase
   (`_Q_MESSAGE_GO` / `_Q_MESSAGE_STOP`); it is never asked to judge
   compliance itself.

That split is a correctness fix, not a style choice. The first version of
this feature combined classification and message into one call, where the
model had to open its own answer with the literal word GO or STOP. Three
different phrasings of that were tried:

- A plain "answer GO or STOP" question → **defaulted to compliant
  regardless of the image**, including a reproducible false GO on a frame
  with no vest anywhere in it (verified against the actual captured JPEG,
  not a staleness artifact).
- A `WORN` / `HELD` / `NONE` multiple-choice → answered **inconsistently
  across repeated calls on the exact same frame**, and wrong both ways
  (called a bare hoodie "WORN", called a visibly-held yellow vest "NONE").
- A "respond in this style: \<example sentence\>" version → reliably
  produced a full, correctly-*shaped* sentence, but still got the actual
  GO/STOP judgment wrong on the no-vest frame from the first bullet.

All three are the same underlying failure: **acquiescence bias**, a
documented small-model tendency to answer a classification-shaped question
from the shape of the question more than the image. The diagnostic that
found this: asking the same model to plainly *describe* what's on the
person's torso (no classification framing at all) got it right every
time — "a black jacket" for the hoodie frame, "a black shirt and a yellow
high-visibility vest" for the held-vest frame. The model's perception was
never the problem. Asking for **counts** turned out to dodge the bias as
reliably as an open description does, and unlike a description, a count is
trivial to parse into a GO/STOP decision — hence the architecture above.
Verified 6/6 correct (3 repeated calls × 2 known-STOP test frames) after
the fix, where the self-classifying version had just failed live.

A smaller, earlier lesson, still relevant if a future prompt needs
open-ended generation: negatively-scoped instructions appended after the
main ask ("describe X... do not mention Y") tend to collapse Qwen2-VL/
moondream2 into a one-word non-answer rather than a full sentence that
respects the exclusion. Front-loading the scope into the main clause (e.g.
*"speaking directly to the person **and only about their safety
vest**..."*) works reliably where the appended-exclusion version doesn't.

## Edge target: Hailo-10H

The AI HAT+ 2 ships with the **Hailo-10H**, not the CNN-only Hailo-8/8L —
worth being explicit about since it changes the deployment plan
substantially:

- **40 TOPS INT4 / 20 TOPS INT8**, **8GB dedicated on-board LPDDR4 RAM**,
  ~2.5W typical, direct DDR interface sized for LLM/VLM weights. SDK:
  Dataflow Compiler, `hailo-ollama`, Model Explorer GenAI.
  ([Hailo product page](https://hailo.ai/products/ai-accelerators/hailo-10h-ai-accelerator/))
- **Officially validated today**: Llama 3.2 1B, Qwen2.5 1.5B,
  DeepSeek-R1-distill-Qwen 1.5B (LLM, via `hailo-ollama`); **Qwen2-VL-2B-Instruct**
  for VLM, via Hailo Apps' `vlm_chat.py` — demoed on this exact Pi 5 +
  Camera Module 3 stack, hence the Track B model choice above.
  ([Hailo blog](https://hailo.ai/blog/bringing-generative-ai-to-the-edge-llm-on-hailo-10h/))
- **Important caveat**: an independent Jan 2026 review benchmarked real
  decode throughput and found it underwhelming for small models — CPU-only
  inference beat the Hailo-10H on tokens/sec for several tested models
  (e.g. Qwen2.5-1.5B-Instruct: 6.7 tok/s on Hailo vs 11.7 tok/s on CPU).
  The reviewer's summary: *"feels more like an AI decelerator than an AI
  accelerator"* for raw throughput. The confirmed real wins were **power
  draw** (7.2–7.6W vs 10.2–10.6W) and **freeing the host CPU/RAM**, not
  guaranteed speed — decode is memory-bandwidth-bound, not TOPS-bound, so
  40 TOPS doesn't translate linearly into tokens/sec.
  ([CNX Software review](https://www.cnx-software.com/2026/01/20/raspberry-pi-ai-hat-2-review-a-40-tops-ai-accelerator-tested-with-computer-vision-llm-and-vlm-workloads/))

**Implication**: don't rely on the chip to make free-text generation fast —
the classification calls already generate as close to zero tokens as
possible (the person/vest counts), and that's deliberate: those two calls
are the ones on the latency-critical path for the GO/STOP decision itself.
The message-phrasing call spends more tokens because the brief specifically
wants the VLM's own sentence, not a template — that's a product decision
made with eyes open about the cost, not an oversight. If Hailo-10H bring-up
shows the message call is the bottleneck on real hardware, swapping it back
to a template lookup (`detectors/messages.py` already has one, currently
only used by the archived `vlm_moondream.py`) is a one-line change in
`vlm_qwen2vl.py`'s `infer()`, not a redesign.

## Benchmark on still images

```bash
uv run python scripts/benchmark.py --images data/samples --backend yolo_hsv
uv run python scripts/benchmark.py --images data/samples --backend vlm_paligemma
# vlm_qwen2vl / vlm_qwen2vl_7b / vlm_moondream are archived — re-register
# in detectors/registry.py's _lazy_register() first (see note above)
```

Name files `xxx__GO.jpg` / `xxx__STOP.jpg` to also get accuracy against a
filename label. Drop test images into `data/samples/`.

## Roadmap

1. **Prototype (this repo, laptop GPU)** — get both tracks running, compare
   latency/accuracy on the RTX 3000 8GB. ✅ both live and usable, with a
   genuine (not templated) VLM message and a verified-correct verdict
   pipeline — see [Prompt design](#prompt-design-vlm-detection) for the
   acquiescence-bias bug that took three iterations to actually fix.
2. **Port to Pi 5 + AI HAT+ 2 (Hailo-10H)** — hardware not yet in hand.
   Bring up `vlm_qwen2vl` via Hailo's `vlm_chat.py` reference path first
   (known-good starting point), then get real latency/power numbers rather
   than projecting from the RTX 3000 — see the Hailo-10H caveat above,
   real decode speed on this chip is not a given.
3. **Fine-tune for a single-token verdict (B200 cluster)** — the task is
   binary classification, not open-ended VQA. Fine-tune Qwen2-VL-2B (same
   architecture family as the Hailo-validated model, to minimize
   compile-compatibility risk later) to answer in ~1 token
   (`0`/`1`/`YES`/`NO`) instead of parsed integer counts, using:
   - public PPE datasets + auto-labeled site-realistic captures (teacher
     VLM labeling at scale, using the B200s' spare capacity for data
     volume rather than just model size)
   - INT4 quantization via Hailo's Dataflow Compiler targeting the
     `hailo-ollama`/GenAI pipeline (the tooling-supported path today)
   - Track A (YOLO) fine-tuned on real vest-labeled data as the hedge —
     Hailo's classic vision toolchain (Model Explorer *Vision*, separate
     from GenAI) is far more mature than the GenAI path, so this is the
     fallback most likely to hit real-time latency if the VLM path
     underwhelms on-chip the way the independent review suggests it might.
4. **Re-benchmark on real hardware** once the AI HAT+ 2 arrives — extend
   `scripts/benchmark.py` with the compiled-for-Hailo backend(s).

## Layout

```
src/icp_safety_vlm/
  detectors/
    base.py            # VestDetector interface, Detection/DetectionResult
    messages.py         # fixed GO/STOP message templates
    yolo_hsv.py          # Track A
    vlm_paligemma.py      # Track B (active): PaliGemmaDetector (On-Prem,
                          #   full precision) + PaliGemma2QuantizedDetector
                          #   (On-Device / Jetson candidate, 4-bit)
    vlm_qwen2vl.py        # archived (not registered) — Qwen2-VL-2B/-7B,
                          #   see README status note + system-design-
                          #   integration.md for why it moved to archived
    vlm_moondream.py       # archived comparison baseline (not registered)
    registry.py              # name -> detector class
  server/          # FastAPI app (Camera/Sentry) + browser UI (templates/index.html)
                   #   Camera supports CAMERA_SOURCE for a remote camera_relay.py
scripts/
  benchmark.py     # latency/accuracy comparison across backends, one at a time
  camera_relay.py  # serves a local webcam over HTTP for a remote GPU box
                    #   with no camera (see Distributed setup above)
data/samples/      # drop test images here
models/            # downloaded YOLO weights (gitignored)
```
