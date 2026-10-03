"""
Live webcam vest detector, Ollama-served VLM, single-call optimized prompt.

Same model registry, SAFETY_PROMPT, and parse_count() as
test_ollama_models_optimized.py -- the only thing that changes here is
the frame source: a live USB webcam instead of a fixed test image or
video file, captured and re-inferred continuously until stopped.

VLM inference is far slower than a classic CV pipeline like
vest_check_temporal_live.py's YOLO backend -- expect roughly 1-5
frames per second at best, not 30fps, depending on the model chosen.
Each loop grabs whatever the webcam's latest frame is right before
calling the model, rather than queuing frames, so the display stays
reasonably current instead of falling further and further behind.

Requires a real X11 display for the cv2.imshow() window (same
requirement as vest_check_temporal_live.py) -- run this from a
terminal inside the xrdp/XFCE session, not a plain SSH shell, or pass
--no-display for a console-only run with no live window.

Usage:
    python3 test_ollama_live_webcam_optimized.py --source 0
    python3 test_ollama_live_webcam_optimized.py --source 1 --no-display
"""

import argparse
import re
import time
from pathlib import Path
from datetime import datetime

import cv2
import ollama

# ============================================================
# Model registry -- same models/entries as the other optimized
# Ollama scripts in this project.
# ============================================================

MODELS = {
    "1": ("Gemma 3 4B", "gemma3:4b"),
    "2": ("Ministral 3 3B Instruct", "ministral-3:3b"),
    "3": ("Moondream", "moondream"),
    "4": ("LLaVA-Phi3", "llava-phi3"),
    "5": (
        "Cosmos Reason 2 2B (community GGUF)",
        "hf.co/apolo13x/Cosmos-Reason2-2B-GGUF:Q4_K_M",
    ),
    "6": ("LLaVA-Llama3", "llava-llama3"),
    "7": ("Granite Vision", "granite3.2-vision"),
    "8": ("BakLLaVA", "bakllava"),
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_DIR = PROJECT_ROOT / "logs"

# ============================================================
# Optimized VLM Prompt -- identical wording to
# test_ollama_models_optimized.py.
# ============================================================

SAFETY_PROMPT = """
Count two things in this image:
1. The total number of clearly visible people.
2. Of those, how many are properly wearing a high-visibility orange,
   yellow, or lime green safety vest on their torso.

Do NOT count as "wearing a vest":
- a vest being held in a hand
- a vest being carried
- a vest resting on a shoulder
- a vest partway through being put on
- a vest lying or hanging nearby but not worn
- ordinary clothing that is not a safety vest
- reflective clothing that is not a high-visibility safety vest

Answer with exactly two lines, nothing else:
PEOPLE: <integer>
VESTS: <integer>
""".strip()

GREEN = (0, 200, 0)
RED = (0, 0, 230)
YELLOW = (0, 200, 255)
BLUE = (255, 160, 0)
WHITE = (255, 255, 255)


# ============================================================
# Logging
# ============================================================

def log_print(message, log_file=None):
    print(message)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


# ============================================================
# Model helper -- same ask()/parse_count() as the other
# optimized scripts, frame passed as in-memory JPEG bytes.
# ============================================================

def ask(model_name: str, frame_bytes: bytes) -> tuple[str, float]:
    start = time.perf_counter()

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": SAFETY_PROMPT,
                "images": [frame_bytes],
            }
        ],
        options={
            "temperature": 0.0,
            "num_ctx": 8192,
        },
    )

    elapsed = time.perf_counter() - start

    return response["message"]["content"].strip(), elapsed


def parse_count(text: str, label: str) -> int:
    if not text:
        return 0
    match = re.search(rf"{label}\s*:\s*(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else 0


# ============================================================
# Overlay drawing -- same label() style as
# vest_check_temporal_live.py, for visual consistency with the
# YOLO track.
# ============================================================

def label(img, text, x, y, color):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y - h - 8), (x + w + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)


def status_color(status):
    if status == "GO":
        return GREEN
    if status == "STOP":
        return RED
    return YELLOW  # NO_PERSON


# ============================================================
# Model selection
# ============================================================

def select_model() -> tuple[str, str]:
    print("Select a model:")
    for key, (display_name, _) in MODELS.items():
        print(f"  [{key}] {display_name}")

    choice = input("Enter number: ").strip()

    if choice not in MODELS:
        raise ValueError(f"'{choice}' is not a valid option.")

    return MODELS[choice]


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index, e.g. 0 or 1")
    ap.add_argument("--no-display", action="store_true",
                     help="console-only, no cv2 live window (use when no X11 display is available)")
    ap.add_argument("--log", default=None, help="log file path; default is a timestamped file under logs/")
    args = ap.parse_args()

    display_name, model_name = select_model()

    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam source: {args.source}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.log:
        log_path = Path(args.log)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = LOG_DIR / f"{timestamp}_live_webcam_{model_name.replace(':', '_').replace('/', '_')}.txt"

    log_file = open(log_path, "w", encoding="utf-8")

    log_print("=" * 60, log_file)
    log_print(f"LIVE WEBCAM VEST DETECTOR (OLLAMA, SINGLE-CALL OPTIMIZED PROMPT)", log_file)
    log_print("=" * 60, log_file)
    log_print(f"Model: {model_name}", log_file)
    log_print(f"Webcam source: {args.source}", log_file)
    log_print(f"Log file: {log_path}", log_file)
    log_print("", log_file)
    log_print("Running warmup inference (absorbs first-call model-load cost)...", log_file)

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Could not read an initial frame from the webcam.")

    ok, buffer = cv2.imencode(".jpg", frame)
    warmup_start = time.perf_counter()
    _ = ask(model_name, buffer.tobytes())
    log_print(f"Warmup inference: {time.perf_counter() - warmup_start:.3f}s", log_file)

    if args.no_display:
        log_print("Running headless (--no-display). Press Ctrl+C to stop.", log_file)
    else:
        log_print("Press Q or ESC in the camera window to stop.", log_file)

    log_print("-" * 60, log_file)

    frame_count = 0
    total_time = 0.0

    try:
        while True:

            # ----------------------------------------------------
            # Grab the latest frame right before inference, not
            # whatever's oldest in the buffer -- keeps the display
            # from drifting further behind real time as inference
            # latency accumulates.
            # ----------------------------------------------------

            ok, frame = cap.read()
            if not ok:
                log_print("Lost webcam frame, stopping.", log_file)
                break

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            frame_count += 1

            response, inference_time = ask(model_name, buffer.tobytes())

            n_people = parse_count(response, "PEOPLE")
            n_vests = min(parse_count(response, "VESTS"), n_people)

            total_time += inference_time
            average = total_time / frame_count

            status = "GO" if (n_people > 0 and n_vests >= n_people) else (
                "NO_PERSON" if n_people == 0 else "STOP"
            )

            log_print(
                f"#{frame_count}  people={n_people} vests={n_vests} "
                f"status={status}  inference={inference_time:.3f}s  "
                f"avg={average:.3f}s  raw={response!r}",
                log_file,
            )

            if not args.no_display:

                color = status_color(status)

                label(frame, f"Status: {status}", 10, 35, color)
                label(frame, f"People: {n_people}  Vests: {n_vests}", 10, 70, BLUE)
                label(
                    frame,
                    f"{model_name}  {inference_time:.2f}s  (avg {average:.2f}s)",
                    10, 105, BLUE,
                )

                cv2.imshow("VLM Vest Detector (live)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

    except KeyboardInterrupt:
        log_print("", log_file)
        log_print("Stopped (Ctrl+C).", log_file)

    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

        log_print("-" * 60, log_file)
        if frame_count > 0:
            log_print(f"Total inferences: {frame_count}", log_file)
            log_print(f"Average inference time: {total_time / frame_count:.3f}s", log_file)
        log_print(f"Log saved to: {log_path}", log_file)

        log_file.close()


if __name__ == "__main__":
    main()
