# System Design: ICP Safety VLM Sentry — On-Prem + On-Device Integration

## Purpose

One "safety vest sentry" application (browser GO!/STOP! banner + one-sentence
model-authored explanation, from a live camera feed), built and benchmarked
as **two deployment solutions** against the same detector interface, so
latency/accuracy/power tradeoffs are measured head-to-head rather than
argued about:

1. **Solution A — On-Prem**: server-grade GPU (brannigan, RTX 4080 16GB)
   runs both the classic-CV track (YOLO+HSV) and the VLM track, viewed from
   a browser on the dev laptop (P16v). This is the accuracy/latency
   ceiling — what's achievable with real GPU headroom — and the primary
   YOLO-vs-VLM comparison rig.
2. **Solution B — On-Device (edge)**: a Jetson Orin Nano Super runs the
   *same application*, same detector interface, standalone — no laptop, no
   brannigan — using a quantized, smaller-footprint variant of the same
   VLM base model. This is the real edge-deployment candidate.

Both solutions share `detectors/base.py`'s `VestDetector` interface and
`scripts/benchmark.py`, so a result from one is directly comparable to the
other, and the same prompt-design lessons (see README's [Prompt
design](README.md#prompt-design-vlm-detection)) transfer.

## Model choice: PaliGemma 2 (3B), replacing Qwen2-VL

Qwen2-VL-2B-Instruct (the prototype's original VLM, see `cdf1f81`) is
Alibaba-origin. This phase of the project requires a **non-CN-origin**
open-source model, so both tracks move to **PaliGemma 2 (`google/paligemma2-3b-*`)**:

- Google/DeepMind origin, Apache-style open weights (Gemma license —
  requires accepting Google's license on Hugging Face before first
  download, same friction as any Gemma-family model).
- **Same base model on both solutions**, per the brief: On-Prem runs it at
  full precision (fp16/bf16), On-Device runs a quantized (INT4/INT8) copy
  of the *same* 3B checkpoint — "smaller" here means footprint via
  quantization, not a different parameter count, which keeps the two
  solutions' outputs comparable (same weights, same training, different
  numeric precision) rather than comparing two genuinely different models.
- Architecturally a better fit for this project's prompting style than
  Qwen2-VL was: PaliGemma is a single-turn image+text-prefix → text
  completion model (no chat template, no "assistant" framing to fight
  against) — the existing count-based prompt design
  (`_Q_PEOPLE`/`_Q_VESTS`/message call) maps onto `"answer en <question>"`
  -style prefixes directly. This may simplify the acquiescence-bias
  mitigation work documented in the README, though that still needs to be
  re-verified empirically against PaliGemma, not assumed.
- Available at 224/448/896px input resolutions — 224px is almost certainly
  the right choice for the counting-style prompts here (person/vest counts
  don't need fine detail), and keeps compute/VRAM down on both ends.

**Open item, not yet decided**: exact quantization scheme for the edge
copy (INT4 via bitsandbytes as done for the archived `vlm_qwen2vl_7b`
track, vs. a Jetson-native path — TensorRT-LLM or ONNX Runtime with a
Jetson-optimized INT8/INT4 export). The Jetson's installed stack
(TensorRT 10.16, cuDNN 9.20, CUDA 13.2 — see [On-Device
environment](#on-device-environment-jetson-orin-nano-super) below) favors
a TensorRT export for real edge throughput, but bitsandbytes-on-transformers
is the faster path to a first working comparison point. Recommend
prototyping with bitsandbytes first (matches the On-Prem code path,
minimal new work) and only moving to a TensorRT export if the throughput
numbers demand it — mirrors this project's existing "prototype first,
optimize once real numbers exist" pattern from the Hailo-10H roadmap.

## Solution A — On-Prem architecture

```
┌─────────────────────────┐        SSH reverse tunnel        ┌──────────────────────────────────────┐
│  P16v laptop (WSL2)      │   (rides existing SSH ACL;       │  brannigan (RTX 4080, 16GB VRAM)      │
│  RTX 3000 Ada 8GB        │    direct Tailscale port is      │                                       │
│                          │    blocked cross-tailnet —       │  uvicorn icp_safety_vlm.server.app    │
│  ┌────────────────────┐ │    see Networking note below)    │    :8010                              │
│  │ webcam (usbipd      │ │                                  │    CAMERA_SOURCE=http://localhost:8100│
│  │ passthrough)        │ │  ssh -R 8100:localhost:8100 ───▶ │    /frame  (via tunnel)               │
│  └─────────┬───────────┘ │                                  │                                       │
│            │             │                                  │  ┌─────────────────────────────────┐  │
│  ┌─────────▼───────────┐ │                                  │  │ Camera (Camera._loop_remote)     │  │
│  │ camera_relay.py      │ │                                  │  │  → Sentry(yolo_hsv)               │  │
│  │  GET /frame :8100    │ │                                  │  │  → Sentry(vlm_paligemma)          │  │
│  └──────────────────────┘ │                                  │  └─────────────────────────────────┘  │
│                          │                                  │                                       │
│  Browser: http://brannigan.taila277ca.ts.net:8010/  ◀───────┼── viewed from here, over Tailscale ───┘
└─────────────────────────┘
```

**Networking note**: `brannigan` and the P16v laptop are on *different*
Tailscale tailnets joined via node-sharing (`brannigan` under
`innovationcentralau@`, laptop under `stanleyoz@`). Confirmed empirically
(2026-09-14 port): `tailscale ping` succeeds between them (ICMP-equivalent
allowed), but a direct TCP connection to an arbitrary port (8100) hangs —
an ACL on one side scopes the shared node to specific ports, SSH (22)
among them. Rather than request an ACL change (organizational, slower,
other admins involved), the working fix is an **SSH reverse tunnel**
(`ssh -R 8100:localhost:8100 -N stanley@brannigan...`) riding over the
already-permitted SSH connection. This tunnel is not currently a persistent
service — see [Operational gaps](#operational-gaps--next-steps).

**Already built and verified working** (as of 2026-09-14, with
`vlm_qwen2vl` as a placeholder for the not-yet-integrated
`vlm_paligemma`):
- `scripts/camera_relay.py` — laptop-side webcam-over-HTTP relay.
- `server/app.py`'s `Camera` class — `CAMERA_SOURCE` env var for pulling
  frames from a remote relay instead of a local device.
- End-to-end verified: `yolo_hsv` (2.8ms/frame, correct verdict),
  `vlm_qwen2vl` (890ms/frame vs. 1.7–2.9s on the 8GB laptop — the extra
  VRAM headroom on the 4080 is a real, measured win), idle-pause/unload
  behavior unaffected by the network split.

**Still to build**: `detectors/vlm_paligemma.py` (replaces
`vlm_qwen2vl.py` as the active on-prem VLM track — `vlm_qwen2vl.py` and
the `vlm_qwen2vl_7b` accuracy-track experiment can move to archived status
the way `vlm_moondream.py` already is, once PaliGemma is verified working,
so the lesson-learned prompt-design writeup isn't lost). Needs its own
prompt design pass — do **not** assume the Qwen2-VL acquiescence-bias fix
(count-based verdict, model only phrases the message) transfers unverified;
re-run the same kind of adversarial testing (known-STOP frames, repeated
calls) documented in the README before trusting it.

## Solution B — On-Device architecture

```
┌────────────────────────────────────────────────────┐
│  Jetson Orin Nano Super (standalone)                │
│  JetPack 7.2.1 (L4T R39.2.1), Python 3.12.3         │
│  7.4GB unified CPU/GPU memory                        │
│  CUDA 13.2, cuDNN 9.20, TensorRT 10.16 (installed    │
│  via SDK Manager over USB-C NCM, 2026-09-14)         │
│                                                       │
│  Pi/USB camera attached directly ─────┐              │
│                                        ▼              │
│  Same icp_safety_vlm server + Camera(local device)   │
│    → Sentry(yolo_hsv)          [hedge/fallback track]│
│    → Sentry(vlm_paligemma_quantized)                 │
│                                                       │
│  Browser: http://<jetson-tailnet-ip>:8010/ — viewed  │
│  from any tailnet machine once the Jetson joins       │
│  Tailscale (not yet done — see below)                 │
└────────────────────────────────────────────────────┘
```

**Current real state of the Jetson** (checked 2026-09-14, mid-flash via
SDK Manager from brannigan over USB-C, JetPack 7.2.1 target-components
install): OS flashed, booted, reachable at `192.168.55.1` over the USB-C
NCM link from brannigan; CUDA 13.2 / cuDNN 9.20 / TensorRT 10.16 confirmed
installed via `dpkg -l`. **Not yet on Tailscale**, **no camera attached
yet**, **this repo not yet cloned onto it**.

**7.4GB usable RAM is the binding constraint** — tighter than brannigan's
16GB VRAM and tighter than even the 8GB laptop, since on a Jetson that RAM
is *shared* between CPU and GPU (no separate VRAM pool), so the model has
to coexist with the OS, camera capture, and the server process in the same
budget. This is the real reason the edge copy of PaliGemma 2 needs
aggressive quantization, not just "smaller for speed" — it may not fit
unquantized at all alongside everything else running.

**To build**:
1. Get the Jetson onto Tailscale (`tailscale up`) so it's reachable
   without brannigan as a USB-C relay — needed for both remote dev and for
   viewing its dashboard from the laptop browser.
2. Attach a camera directly (the edge target is standalone by design — no
   relay, unlike Solution A).
3. Clone this repo, `uv sync` (confirm `uv` + Python 3.12 compatibility on
   L4T aarch64 — not yet verified, first-time-per-package wheel
   availability on aarch64 is the main risk, especially for
   `bitsandbytes` if that's the chosen quantization path).
4. `detectors/vlm_paligemma_quantized.py` (or a quantization-config
   parameter on the same `PaliGemmaDetector` class, mirroring how
   `QwenVL7BDetector` subclassed `QwenVLDetector` for a different
   precision — same pattern applies here).
5. Re-run `scripts/benchmark.py` on the Jetson directly for real
   latency/power numbers — same principle as the existing Hailo-10H
   roadmap note: project nothing, measure on the real target.

## Shared interface (unchanged by this work)

Both solutions plug into the existing architecture without modification:

- `detectors/base.py`'s `VestDetector` — `warmup()` / `infer()` /
  `unload()` / `timed_infer()`. New detectors implement this and register
  in `detectors/registry.py`; nothing else needs to know which solution
  it's running on.
- `server/app.py`'s `Camera`/`Sentry` — local or `CAMERA_SOURCE`-remote
  capture, idle-pause + unload for GPU/memory contention, unchanged
  between solutions.
- `scripts/benchmark.py` — the tool for **reportable** numbers on either
  machine; the live server (either solution) is for demoing, not clean
  benchmarking, per the existing README caveat.

## Benchmark plan (both solutions)

| Axis | On-Prem (brannigan, fp16) | On-Prem (brannigan, 4-bit) | On-Device (Jetson Orin Nano Super) |
|---|---|---|---|
| yolo_hsv latency | ✅ measured: 2.8ms | n/a | not yet measured |
| VLM latency | ✅ measured: ~295ms/frame (`vlm_paligemma`) | ✅ measured: ~318ms/frame — **slower**, not faster (see below) | not yet measured (`vlm_paligemma_quantized`) |
| VLM VRAM/memory | ✅ measured: 6.08GB | ✅ measured: 2.62GB (57% less — the actual point of this variant) | not yet measured |
| VLM accuracy | ✅ verified live: correct on worn (both colors)/off/held-not-worn, 37/37 across a stability pass — see README status note | ✅ identical output to fp16 on every test frame (expected — same weights, lower precision) | not yet measured |
| Power draw | not applicable (shared server) | not applicable | key edge metric — same Hailo-10H-review lesson applies: TOPS/params don't predict tokens/sec or watts, measure on real hardware |

**Real finding (2026-09-14, controlled back-to-back measurement on the
same frame)**: on brannigan, 4-bit quantization is a pure memory-footprint
trade, not a speed or accuracy win. `vlm_paligemma_quantized` uses 57%
less VRAM but is *slightly slower* (~318ms vs. ~295ms) with identical
output — bitsandbytes falls back to a slower dequant kernel for this
model's layer dimensions
(`inner dimension (4304) is not aligned for fast kernel with blocksize=64`),
and brannigan's 16GB has no memory pressure for quantization to relieve
in the first place. **This does not predict the Jetson result** — different
bitsandbytes build (ARM/Ampere vs. x86/Ada), and the Jetson's 7.4GB
*is* genuinely memory-constrained, which is the actual scenario
quantization is meant to help — measure there directly rather than
assume either a repeat of this finding or its opposite. See
`vlm_paligemma.py`'s `PaliGemma2QuantizedDetector` docstring for the full
numbers.

`scripts/benchmark.py` already supports `--backend` selection and
GO/STOP-labeled filename accuracy scoring (`data/samples/xxx__GO.jpg`) —
extend it with per-solution result tagging (e.g. an output column for
which machine ran it) once both are live, so results from the two
solutions can sit in one comparison table.

## Operational gaps / next steps

Not yet solved, called out explicitly so they aren't lost:

1. **On-Prem's SSH tunnel + camera relay are foreground/manual
   processes**, not systemd services — they don't survive a terminal
   closing or a reboot. Needs a supervised setup (systemd user units, or
   equivalent) before this is a standing demo rather than a
   manually-started one.
2. **Jetson is not yet on Tailscale** and has no camera attached — first
   two concrete steps for Solution B.
3. **PaliGemma's Gemma license** must be accepted on Hugging Face
   (per-account, one-time) before either machine can download it —
   confirm this is done before scripting an unattended first-run
   download, or it'll fail with a 403 instead of a clear "accept the
   license" message.
4. **Quantization path for the Jetson is undecided** (bitsandbytes vs.
   TensorRT export) — see [Model choice](#model-choice-paligemma-2-3b-replacing-qwen2-vl)
   above; needs a decision once `bitsandbytes` aarch64 wheel availability
   on JetPack 7.2.1 is confirmed one way or the other.
5. **PaliGemma prompt design is unverified** — do not assume the
   Qwen2-VL acquiescence-bias fix transfers; re-test.
