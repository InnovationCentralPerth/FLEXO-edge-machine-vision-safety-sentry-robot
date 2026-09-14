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
│  icp-nano-flexo (Jetson Orin Nano Super, standalone)│
│  JetPack 7.2.1 (L4T R39.2.1), Python 3.12.3, on      │
│  Tailscale. 7.4GB unified CPU/GPU memory.            │
│  CUDA 13.2, cuDNN 9.20, TensorRT 10.16 (installed    │
│  via SDK Manager over USB-C NCM, 2026-09-14)         │
│                                                       │
│  USB camera attached directly (/dev/video0) ──┐      │
│                                                ▼      │
│  Same icp_safety_vlm server (not yet run as a live   │
│  server here — tested via ad-hoc script so far)      │
│    → Sentry(yolo_hsv)          [hedge/fallback track]│
│    → Sentry(vlm_paligemma)             [fp16]        │
│    → Sentry(vlm_paligemma_quantized)   [4-bit]       │
│                                                       │
│  Browser: http://icp-nano-flexo.<tailnet>:8010/ once │
│  the server is actually run here (not yet done)      │
└────────────────────────────────────────────────────┘
```

**Current real state of the Jetson (updated 2026-09-14)**: on Tailscale
as `icp-nano-flexo`, USB camera attached (`/dev/video0`), repo present
(rsync'd from brannigan's clone rather than a fresh `git clone` — no
GitHub SSH key set up on this device yet, so `git pull` there won't work
until that's added; re-sync via rsync from brannigan for now), `uv`
installed, HF token copied from brannigan's cache (same account, license
already accepted). CUDA 13.2 / cuDNN 9.20 / TensorRT 10.16 confirmed
installed via `dpkg -l` during the SDK Manager flash. **Not yet done**:
running as a live server (only ad-hoc script testing against saved
frames so far — see results below), systemd/persistent service setup.

**7.4GB usable RAM is the binding constraint** — tighter than brannigan's
16GB VRAM and tighter than even the 8GB laptop, since on a Jetson that RAM
is *shared* between CPU and GPU (no separate VRAM pool), so the model has
to coexist with the OS, camera capture, and the server process in the same
budget.

**Real finding: generic PyPI `torch` (the same `2.13.0+cu130` wheel used
on brannigan) installs and functionally runs on the Jetson's Orin GPU,
but is NOT an officially supported build.** PyTorch itself warns at
import: `No published PyTorch CUDA builds for release 2.13.0+cu130
support this GPU` — Orin is compute capability 8.7, and this wheel was
only compiled for other SMs (8.0/9.0/10.0/11.0/12.0). It works via
forward-compatible PTX JIT compilation, confirmed with a real matmul, not
just the `torch.cuda.is_available()` flag. This has a measurable, real
cost: **the first inference call after warmup pays a one-time ~3.5-7s
kernel-compilation tax** (confirmed by re-running the same frame after
warmup: 7235ms first call → 1382ms, 1357ms, 1359ms on repeats). A live
server should pre-warm with a dummy inference at startup rather than
serve this delay to the first real viewer. This is also the concrete,
measured version of "why TensorRT" — a properly Jetson-targeted build
(native SM 8.7 kernels, or a TensorRT export) should remove both the JIT
tax and likely improve steady-state throughput, though that's not yet
measured, only implied by the mismatch warning.

**Results (2026-09-14, ad-hoc script against the same saved test frames
used on brannigan — not yet a live server test)**:

| Backend | Verdict accuracy | Steady-state latency | Memory |
|---|---|---|---|
| `yolo_hsv` | 4/4 correct (2-person-both-worn, 1-person-worn ×2 colors, 1-person-held-not-worn) | 25-77ms | n/a |
| `vlm_paligemma` (fp16) | 4/4 + 3/3 repeat, multi-person fix holds (`GO`, n=2, 3/3 identical repeats) | ~1.35-1.4s | not measured precisely, but fp16 weights alone are ~6GB of the 7.4GB total pool |
| `vlm_paligemma_quantized` (4-bit) | 3/3 correct | ~1.92s (**slower than fp16, same pattern as brannigan — confirmed to transfer, resolving the earlier "may not transfer" open question**) | 2.61GB (matches brannigan's 2.62GB almost exactly) |

**The memory-vs-speed tradeoff is genuinely different here than on
brannigan.** On brannigan, quantization has no upside (16GB has zero
memory pressure to relieve) and a real speed cost — fp16 is strictly
better there. On the Jetson, fp16's ~6GB against a 7.4GB *total* pool
leaves only ~1.4GB for the OS, camera capture, and the server process —
genuinely tight. The quantized 2.61GB leaves ~4.8GB of headroom despite
being ~40% slower. Whether that trade is worth it depends on how tight
the full running system (not just the model) actually is in practice —
worth re-measuring memory headroom with the whole server (camera +
model) running, not just the model in isolation, before deciding which
variant is the real On-Device default.

**Systemd service**: `deploy/jetson/icp-safety-sentry.service` (installed
at `/etc/systemd/system/` on the Jetson, `enable`d for
`multi-user.target`) — source of truth for how it's actually deployed
there; re-copy and `systemctl daemon-reload` after editing rather than
hand-editing the installed copy.

**Live server verified working end-to-end (2026-09-14)**, real Jetson
camera, real server code path (not the ad-hoc script above): both
`yolo_hsv` and `vlm_paligemma` ran live via `uvicorn ... --port 8010`
with the local `CAMERA_INDEX` (no `CAMERA_SOURCE` relay — camera attached
directly, per the standalone design). Idle-pause/unload behaved as
expected across the multi-hop polling used to test this (brannigan → 
Jetson SSH round-trips exceed `IDLE_TIMEOUT_S`, so the Sentry idled and
reloaded between checks — consistent with the documented design, not a
bug).

**Real finding: `yolo_hsv` gives a false STOP on this camera** — a
person clearly wearing an orange hi-vis vest was marked "NO VEST",
reproducible across 3 repeated polls. On the identical live scene,
`vlm_paligemma` correctly said `GO`. The HSV color thresholds in
`detectors/yolo_hsv.py` were tuned against the dev laptop's webcam;
this is real evidence they don't generalize to a different camera's
color calibration/white balance/exposure — a genuine portability
limitation of the classic-CV track, not a Jetson-specific bug (the same
thresholds would likely misfire on brannigan too if fed frames from this
same USB camera). Worth re-tuning the HSV thresholds per-camera, or
re-deriving them from a broader thresholding approach, before treating
`yolo_hsv` as reliable on the Jetson's actual camera — currently it
is not.

**Real finding: the server is NOT autostarted and got OOM-killed
(2026-09-14).** After a power cycle, the dashboard was unreachable —
not because it failed to autostart (it was never set up to; it's still
a foreground `nohup` process, see below), but because the *previous*
run had already been killed by the kernel OOM killer:
```
oom-kill: task=uvicorn pid=6848
Out of memory: Killed process 6848 (uvicorn) anon-rss:6522544kB
```
(confirmed via `sudo dmesg -T`). Cause: the server log showed `yolo_hsv`
and `vlm_paligemma` (fp16, ~6GB) being polled concurrently — two
different tailnet source IPs, likely two open browser tabs/dropdown
switches — which keeps *both* Sentries resident at once (the existing
idle-pause/unload design only unloads a backend once *it specifically*
stops being polled, it doesn't cap total concurrent backends). On
brannigan's 16GB that's harmless; on the Jetson's 7.4GB *total* shared
pool — also running a full GNOME desktop, not headless — 6GB+ for one
VLM backend plus anything else pushed it over the edge and the kernel
killed the whole process, losing both backends at once.

**Mitigated (2026-09-14): the Jetson's systemd service now sets
`DETECTOR_BACKEND=vlm_paligemma_quantized`**, so the landing page (no
`?backend=` param) defaults to the 2.61GB quantized track instead of
fp16's ~6GB — real headroom for `yolo_hsv` to run concurrently without
approaching the 7.4GB ceiling. **This is a mitigation, not a structural
fix**: `vlm_paligemma` (fp16) is still selectable from the dropdown or
via `?backend=vlm_paligemma`, and choosing it manually still carries the
same OOM risk as before if run concurrently with anything else. Capping
total concurrent resident backends in the Sentry architecture itself
(so the server refuses/unloads-oldest rather than letting memory grow
unbounded) would be the structural fix, still not done.

**Second real finding, still open: `yolo_hsv` latency stayed elevated
(~350-384ms) even after a restart, well above the earlier ~25-77ms
baseline.** Root cause attempt: the Jetson had booted into `nvpmodel`'s
config-file default of `25W` power mode (not `MAXN_SUPER`, which was
likely active — manually set, not persisted — during the earlier same-day
benchmarking). Tried `sudo nvpmodel -m 2` (MAXN_SUPER) + `sudo
jetson_clocks`: CPU clocks did increase (1.344GHz → 1.728GHz), but
`nvpmodel -m 2` itself errored (`Error opening
.../17000000.gpu/devfreq_dev/available_frequencies`) — a real driver/sysfs
quirk on this JetPack R39.2.1 build, not a typo or wrong command. GPU
clocks likely stayed capped despite the CPU fix, which would explain why
inference latency (GPU-bound) didn't recover even though the power mode
command "ran". Not yet resolved — needs either a full reboot after
setting the mode (rather than a live `-m` switch), or investigating why
the GPU devfreq sysfs path is missing/inaccessible on this build.

**Third real finding: the GPU driver wedged after the OOM-kill/restart
cycle, failing CUDA init for every process** (`libnvrm_gpu.so:
NvRmGpuLibOpen failed, error=4`), confirmed in the actual systemd
service's own log, not just ad-hoc testing. Methodically ruled out both
of the obvious suspects before concluding this: (1) environment
variables — reproduced the failure with a stripped-down `env -i`
environment, then proved it was NOT the cause by restoring every
variable from a working interactive session (`XDG_RUNTIME_DIR`,
`DBUS_SESSION_BUS_ADDRESS`, etc.) inside the same `env -i` wrapper and
still getting the identical failure; (2) group membership / device
cgroup policy — the actual running service process has identical
supplementary groups to an interactive login (`video`, `render`, etc. —
checked via `/proc/<pid>/status`), and `systemctl show` confirms no
device sandboxing (`DevicePolicy=auto`, `PrivateDevices=no`, all
`Protect*=no`). What actually correlates: every failing test happened
*after* today's OOM-kill/restart cascade (10 restarts); the original
successful CUDA verification (the matmul test, and the working live
server test) both happened *before* that cascade began. This points to
the Tegra NVRM userspace driver being left in a bad state by repeated
`SIGKILL`s during CUDA init — a known class of embedded-GPU driver
issue — not a systemd config problem. **Fixed by rebooting** (the
standard remedy for a wedged embedded GPU driver state); re-verify CUDA
works post-reboot before trusting any Jetson benchmark run after this
point in the log.

**Also fixed while investigating**: the deployed systemd unit had
`StartLimitIntervalSec`/`StartLimitBurst` in the `[Service]` section,
where systemd silently ignores them (confirmed via `journalctl`:
`Unknown key name 'StartLimitIntervalSec' in section 'Service',
ignoring`) — they belong in `[Unit]`. The crash-loop protection never
actually applied during today's incident (10 restarts, zero throttling).
Corrected in `deploy/jetson/icp-safety-sentry.service`.

**Next steps**:
1. ~~Run the actual FastAPI server on the Jetson~~ — done, see above.
2. ~~Fix the OOM-kill risk~~ — **mitigated**: Jetson now defaults to
   `vlm_paligemma_quantized`, see finding above. **Still open**: capping
   total concurrent resident backends in the Sentry architecture itself,
   the structural fix — manually selecting fp16 `vlm_paligemma` still
   carries the same risk.
3. **Resolve the GPU devfreq/power-mode issue** so the Jetson reliably
   runs at its real max performance rather than the `25W` boot default —
   try a reboot after `nvpmodel -m 2` instead of a live switch, or
   investigate the missing `.../17000000.gpu/devfreq_dev/` sysfs path.
4. Fix or re-tune `yolo_hsv`'s HSV thresholds for the Jetson's actual
   camera (see finding above) — currently gives false negatives here.
5. Set up a proper GitHub credential on this device (deploy key or the
   user's own key) so `git pull` works directly instead of rsync-from-brannigan.
6. Measure real end-to-end memory headroom with the full server + camera
   running, to settle the fp16-vs-4-bit question above with real numbers
   instead of a plausible-sounding tradeoff — now more urgent given the
   confirmed OOM-kill above.
7. The TensorRT optimization work (per the user's stated goal: "benchmark
   performance/latency, then optimize for model size + Jetson's
   TensorRT") — not started. The unsupported-build finding above is the
   concrete evidence for why this matters, not just a roadmap aspiration.
8. ~~Systemd/persistent service setup~~ — **done for the Jetson
   (2026-09-14)**: `/etc/systemd/system/icp-safety-sentry.service`,
   `Restart=on-failure` (covers OOM-kills, which are SIGKILL), enabled
   for `multi-user.target` (survives reboot), `StartLimitBurst=8` per
   300s to avoid a crash-loop if the OOM condition (item 2) recurs
   immediately. **Not yet done on brannigan** (both the `:8010` and
   `:8020` server processes) or the camera relays/tunnels (laptop, demo
   client `icp-gmk-01`) — deferred, not forgotten; still foreground
   `nohup` processes there.

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
| VLM accuracy | ✅ verified live: 37/37 single-person stability pass + multi-person fix re-verified on saved frames (not yet re-verified live with two people) — see README's "On-Prem results log" | ✅ identical output to fp16 on every test frame (expected — same weights, lower precision) | not yet measured |
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

**Second real finding (2026-09-14): a larger same-family model
(PaliGemma2-10B-mix, 8-bit — required to fit brannigan's 16GB, since 10B
at bf16 is ~20GB) is worse on every measured axis, not better.** VRAM
10.75GB vs. 3B's 6.08GB; latency ~1.18s/frame vs. ~295ms (~4x slower);
and — the disqualifying one — it answers the vest-count question
**wrong** (`"0"`) on a frame with one person clearly, properly wearing a
yellow vest, despite correctly answering direct questions about the same
frame ("yes" to wearing a safety vest, "yellow" for its color). This
isn't the 3B word-vs-digit parsing bug (10B answered a literal digit);
the counting capability itself regressed for this prompt shape between
the 3B-mix and 10B-mix checkpoints. Decided (2026-09-14) not to pursue a
10B-specific prompt redesign — kept as a documented negative result, see
`vlm_paligemma.py`'s `PaliGemma10BDetector` docstring for the full
diagnostic. **3B remains the confirmed better choice** for this task on
every axis measured so far.

**Third real finding (2026-09-14, user-reported): the original vest-count
prompt undercounted with 2+ people**, producing a false STOP on a frame
with 2 people both properly wearing vests (answered "1", not "2").
Root-caused to two independently-fragile clauses (a color list, and the
word "properly") rather than "compound questions are fragile in general"
— see README's "On-Prem results log" for the before/after prompt text and
`vlm_paligemma.py`'s module docstring for the full diagnostic. Fixed and
re-verified through the actual deployed detector on saved test frames;
still needs a live two-person re-check when available.

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
5. **PaliGemma prompt design has real, found-by-testing history, not a
   clean transfer of Qwen2-VL's fix** — three separate issues found and
   fixed live so far: message generation needed a describe+template
   hybrid (compound generation instructions get refused outright), count
   answers can come back as words not digits (`_parse_int` now handles
   both), and the original `_Q_VESTS` phrasing undercounted with 2+
   people (root-caused to a color list and the word "properly", each
   independently fragile; fixed 2026-09-14 — see `vlm_paligemma.py`'s
   module docstring). Still not adversarially tested for a *mixed*
   multi-person frame (some compliant, some not) or 3+ people — don't
   assume the fix generalizes past what's been directly tested.
