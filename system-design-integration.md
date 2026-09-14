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
| `vlm_paligemma_quantized` (4-bit) | 3/3 correct | ~1.92s (**slower than fp16, same pattern as brannigan — confirmed to transfer, resolving the earlier "may not transfer" open question**) | ~5.6-6GB real process footprint (cgroup `MemoryCurrent`, live server) — 2.61GB was only the tensor-allocator figure, see correction below |

**The memory-vs-speed tradeoff is genuinely different here than on
brannigan.** On brannigan, quantization has no upside (16GB has zero
memory pressure to relieve) and a real speed cost — fp16 is strictly
better there. On the Jetson, fp16's ~6GB+ against a 7.4GB *total* pool
leaves very little for the OS, camera capture, and the server process —
this is what actually triggered the confirmed OOM-kill. Quantized's real
~5.6-6GB footprint leaves a real but modest ~2.3GB available (confirmed
via `free -h` on the live service) despite being ~40% slower — better
than fp16, not a generous margin. Whether that trade is worth it depends
on how tight the full running system (not just the model) actually is —
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

**Superseded (2026-09-14): first mitigated with a default, then fixed
structurally.** First attempt set
`DETECTOR_BACKEND=vlm_paligemma_quantized`, changing only the *default*
landing backend — `vlm_paligemma` (fp16) remained selectable via the
dropdown or `?backend=`, so the OOM risk wasn't actually closed, just
made less likely by default. Verifying it also surfaced a correction:
the earlier "2.61GB" quantized-model figure was
`torch.cuda.memory_allocated()` (tensor-allocator bookkeeping only) —
the real cgroup-measured footprint (`systemctl show -p MemoryCurrent`)
was **~5.6-6GB**, leaving only ~2.3GB available (`free -h`), not the
~4.8GB headroom first assumed. Quantized alone wasn't the generous
margin it looked like.

**Structural fix**: `server/app.py` gained a `LOCKED_BACKEND` env var —
when set, every route resolves to that one backend regardless of the
dropdown or `?backend=` query param (not just changing what loads by
default). The Jetson's systemd service now sets
`LOCKED_BACKEND=vlm_paligemma` (fp16, per user decision — chosen over
locking to quantized since quantized's real memory win turned out
smaller than first measured, it's slower, and offers no accuracy
upside). With only one backend ever selectable, only one Sentry can
ever be resident — the concurrent-multi-backend OOM cause is now
structurally impossible on this deployment, not just less likely.
Re-evaluate if fp16-alone ever approaches the 7.4GB ceiling by itself
(GNOME desktop overhead + fp16 weights, no second backend needed to
trigger it).

**That re-evaluation trigger fired the same day, live.** User reported
the dashboard "stuck at STOP." Caught in the act via SSH: `vlm_paligemma`
(fp16) at **6.9GB RSS / 7.3-7.4GB total used, 91MB free, 919MB in
swap** — not a leak, this was steady state under normal active viewing
(idle-pause only kicks in after `IDLE_TIMEOUT_S=8` with nobody polling;
someone was actively looking at the dashboard). A `systemctl restart`
briefly cleared it (5.5GB available immediately after) but within
seconds of resuming active serving it climbed straight back to
7.3GB used / 1.5GB swap — confirming steady-state, not transient. Under
that swap pressure the inference loop stalls, so the browser keeps
showing the last successfully-rendered frame/verdict — that's the
"stuck" symptom, not a code regression (`_annotate()`'s whole-frame-box
skip was verified unchanged and correct; the bounding box the user saw
was on that same frozen stale frame from before the stall, not newly
drawn).

**Re-evaluated as asked, switched `LOCKED_BACKEND` to
`vlm_paligemma_quantized`.** Measured under the same conditions
(sustained active polling, post-warmup steady state):

| | fp16 (`vlm_paligemma`) | quantized (`vlm_paligemma_quantized`) |
|---|---|---|
| Memory, steady state | 6.9GB RSS / 7.3-7.4GB used, **91MB free**, swap climbing to 1.5GB | 4.7GB used, **2.7GB available**, swap flat at ~294MB |
| Latency (single-question, NO PERSON case) | ~505-513ms | ~745-764ms (~45% slower) |

This flips the earlier trade-off: quantized's memory win is real and
now the deciding factor (2.7GB stable headroom vs. fp16 actively
swapping), even though it's still ~45% slower with no accuracy upside,
as before. `LOCKED_BACKEND=vlm_paligemma_quantized` deployed and
verified live on the Jetson. Not yet done: switching the Jetson off
`graphical.target` to headless (`multi-user.target`) — the dashboard is
already viewed remotely over Tailscale, so the resident GNOME
desktop's ~140MB RSS (gdm/gnome-shell/Xorg/pipewire) is pure overhead;
user deferred this for now in favor of the backend swap above.

**Follow-up, same day: headless switch also applied.** `ssh/tailscaled
/docker/icp-safety-sentry` all confirmed independently under
`multi-user.target` (not pulled in by `graphical.target`), so this was
safe to do without losing remote access. `sudo systemctl set-default
multi-user.target` (persists the choice across reboots) followed by
`sudo systemctl isolate multi-user.target` (applies it immediately, no
reboot needed). Verified: SSH session survived, `icp-safety-sentry`
stayed active throughout with no interruption, gdm/gnome-shell/Xorg
processes confirmed gone (`ps aux` count: 0), swap usage dropped
291MB → 119MB immediately. The Jetson now boots headless from here on;
the dashboard is unaffected since it was already only ever viewed
remotely over Tailscale, never on this device's own screen.

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
2. ~~Fix the OOM-kill risk~~ — **done, structurally**: `LOCKED_BACKEND`
   env var added to `server/app.py`; Jetson's systemd service sets
   `LOCKED_BACKEND=vlm_paligemma`, making concurrent multi-backend
   residency impossible on this deployment. See finding above.
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

   **Investigated 2026-09-14, on the actual device**: the user pointed at
   a claimed PaliGemma2-3B W4A16/INT4 ~1.9GB figure via "Jetson AI Lab
   NanoLLM container". Cloned both `dusty-nv/jetson-containers` and
   `dusty-nv/NanoLLM` directly onto the Jetson and searched — **zero
   PaliGemma references in either repo's actual source.** The only
   PaliGemma-adjacent package (`packages/vlm/gemma_vlm`) has moved away
   from it: its own test script comment reads "Changed from PaliGemma",
   now defaults to `google/gemma-3-4b-it`, and — tellingly — depends on
   `transformers`+`bitsandbytes`, the same stack this project already
   uses, not MLC/TensorRT. `nano_llm` itself does bundle real optimized
   engines (`mlc`, `tensorrt`, `torch2trt`) — architecturally the right
   direction — but its newest published container image is
   `r36.3.0` (2024-08-26), two JetPack generations behind this Jetson's
   R39.2.1; no pre-built image exists for this device, and building the
   8-12GB image stack from source against an untested 2-generation-newer
   JetPack is a real, unbounded risk, not a quick win. Conclusion: the
   specific claimed path isn't currently substantiated against what's
   actually in these repos — worth re-checking if dusty-nv ships an
   R39-compatible build, or if a PaliGemma NanoLLM integration lands
   later, but not actionable today as described.
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

1. ~~On-Prem's SSH tunnel + camera relay are foreground/manual
   processes~~ — **done for brannigan's side (2026-09-14)**: see
   `deploy/brannigan/` — `icp-safety-sentry-main.service` (:8010),
   `icp-safety-sentry-demo.service` (:8020), and
   `icp-camera-tunnel-demo.service` (the brannigan-initiated forward
   tunnel to icp-gmk-01). All three `enable`d, survive a brannigan
   reboot. **Real remaining gap**: `:8010`'s camera source is the P16v
   dev laptop's *reverse* tunnel, initiated from the laptop, not
   brannigan — a personal WSL2 machine isn't a persistent server, so
   that side still needs `scripts/camera_relay.py` + the reverse tunnel
   command re-run by hand after either machine restarts. Only the demo
   pipeline (brannigan-initiated tunnel) recovers fully unattended.
   `icp-gmk-01`'s own `camera_relay.py` process is also still
   foreground/manual — a follow-up on that third machine, not brannigan.
2. **Jetson is not yet on Tailscale** and has no camera attached — first
   two concrete steps for Solution B. *(Superseded — see "Solution B"
   above: the Jetson has been on Tailscale with a camera attached and
   its own systemd service since partway through this project.)*
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
