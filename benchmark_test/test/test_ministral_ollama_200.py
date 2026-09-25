import re
import time
from pathlib import Path
from datetime import datetime

import ollama

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "ministral-3:3b"

IMAGE_PATH = Path("images/test.jpg")

LOG_DIR = PROJECT_ROOT / "logs"

# Test name used in the log filename
LOG_FILENAME = "ministral_ollama_image"

# Number of repeated inferences to run on the *same* loaded image.
# Set to None to run indefinitely (Ctrl+C to stop).
MAX_INFERENCES = 200

# Same count-based prompt shape used throughout this project --
# direct yes/no or GO/STOP classification prompts are known to
# trigger acquiescence bias on small VLMs (see project notes).
PEOPLE_PROMPT = (
    "How many people are visible in this image? "
    "Answer with just a number."
)

VEST_PROMPT = (
    "How many people in this image are properly wearing a "
    "high-visibility orange, yellow, or lime green safety vest "
    "on their torso?\n\n"
    "Do NOT count:\n"
    "- a vest being held in a hand\n"
    "- a vest being carried\n"
    "- a vest resting on a shoulder\n"
    "- a vest partway through being put on\n"
    "- a vest lying or hanging nearby but not worn\n"
    "- ordinary clothing that is not a safety vest\n\n"
    "Answer with just a number."
)


# ============================================================
# Logging
# ============================================================

def log_print(message, log_file):
    """Print message to terminal and save it to the log file."""
    print(message)

    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


# ============================================================
# Model helper
# ============================================================

def ask(image_path: Path, prompt: str) -> tuple[str, float]:
    """Send one image + text prompt to the model. Returns (answer, seconds)."""

    start = time.perf_counter()

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }
        ],
        options={
            "temperature": 0.0,  # deterministic, matches every other detector in this project
        },
    )

    elapsed = time.perf_counter() - start

    answer = response["message"]["content"].strip()

    return answer, elapsed


def parse_number(text: str) -> int:
    """Best-effort parse of a numeric answer; falls back to 0 rather than
    raising if the model returns something unexpected."""
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else 0


# ============================================================
# Main
# ============================================================

def main():

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = LOG_DIR / f"{timestamp}_{LOG_FILENAME}.txt"

    with open(log_path, "w", encoding="utf-8") as log_file:

        log_print("=" * 60, log_file)
        log_print(
            "MINISTRAL-3-3B (OLLAMA) + STATIC IMAGE "
            "REPEATED-INFERENCE TEST",
            log_file
        )
        log_print("=" * 60, log_file)

        log_print(f"Log file: {log_path}", log_file)
        log_print(f"Model: {MODEL_NAME}", log_file)
        log_print(f"Image path: {IMAGE_PATH}", log_file)

        if not IMAGE_PATH.is_file():
            raise FileNotFoundError(
                f"Image not found:\n{IMAGE_PATH}\n"
                f"Update IMAGE_PATH at the top of this script, or "
                f"place a test image at that path."
            )

        # ----------------------------------------------------
        # Throwaway warmup inference
        #
        # Absorbs first-call model-load cost (Ollama loads the
        # model into memory on first request if not already
        # resident) outside the timed loop -- same rationale as
        # every other test script in this project.
        # ----------------------------------------------------

        log_print(
            "Running warmup inference "
            "(absorbs first-call model-load cost)...",
            log_file
        )

        warmup_start = time.perf_counter()

        _ = ask(IMAGE_PATH, PEOPLE_PROMPT)

        warmup_time = time.perf_counter() - warmup_start

        log_print(
            f"Warmup inference: {warmup_time:.3f}s "
            f"(excluded from timing stats below)",
            log_file
        )

        log_print("", log_file)
        log_print("Press Ctrl+C to stop.", log_file)
        log_print("-" * 60, log_file)

        frame_count = 0
        total_time = 0.0

        try:

            while True:

                frame_count += 1

                log_print("", log_file)
                log_print("=" * 60, log_file)
                log_print(f"INFERENCE #{frame_count}", log_file)
                log_print("=" * 60, log_file)

                # -------------------------------------------------
                # 1. Count people
                # -------------------------------------------------

                people_answer, people_time = ask(IMAGE_PATH, PEOPLE_PROMPT)
                n_people = parse_number(people_answer)

                # -------------------------------------------------
                # 2. Count vests
                # -------------------------------------------------

                vest_answer, vest_time = ask(IMAGE_PATH, VEST_PROMPT)
                n_vests = min(parse_number(vest_answer), n_people)

                inference_time = people_time + vest_time
                total_time += inference_time

                # -------------------------------------------------
                # Result
                # -------------------------------------------------

                status = "GO" if (n_people > 0 and n_vests >= n_people) else (
                    "NO_PERSON" if n_people == 0 else "STOP"
                )

                log_print("", log_file)
                log_print("-" * 60, log_file)

                log_print(
                    f"People: {n_people} (raw: '{people_answer}')",
                    log_file
                )
                log_print(
                    f"Vests: {n_vests} (raw: '{vest_answer}')",
                    log_file
                )
                log_print(f"Status: {status}", log_file)

                log_print(
                    f"  people call: {people_time:.3f}s  "
                    f"vest call: {vest_time:.3f}s",
                    log_file
                )

                log_print(
                    f"Inference time: {inference_time:.3f}s",
                    log_file
                )

                average = total_time / frame_count

                log_print(
                    f"Average inference: {average:.3f}s",
                    log_file
                )

                log_print("-" * 60, log_file)

                # -------------------------------------------------
                # Stop automatically at MAX_INFERENCES
                # -------------------------------------------------

                if (
                    MAX_INFERENCES is not None
                    and frame_count >= MAX_INFERENCES
                ):

                    log_print("", log_file)
                    log_print("", log_file)
                    log_print("=" * 60, log_file)

                    log_print(
                        f"TARGET REACHED ({MAX_INFERENCES} inferences)",
                        log_file
                    )

                    log_print("=" * 60, log_file)

                    log_print(f"Total inferences: {frame_count}", log_file)
                    log_print(
                        f"Average inference time: {average:.3f}s",
                        log_file
                    )
                    log_print(
                        f"Total inference time: {total_time:.3f}s",
                        log_file
                    )

                    break

        # ---------------------------------------------------------
        # Ctrl+C
        # ---------------------------------------------------------

        except KeyboardInterrupt:

            log_print("", log_file)
            log_print("", log_file)
            log_print("=" * 60, log_file)
            log_print("TEST STOPPED", log_file)
            log_print("=" * 60, log_file)

            if frame_count > 0:

                average = total_time / frame_count

                log_print(f"Total inferences: {frame_count}", log_file)
                log_print(
                    f"Average inference time: {average:.3f}s",
                    log_file
                )
                log_print(
                    f"Total inference time: {total_time:.3f}s",
                    log_file
                )

        # ---------------------------------------------------------
        # Cleanup
        # ---------------------------------------------------------

        finally:

            log_print("", log_file)
            log_print(
                "Done. (Ollama server keeps running in the "
                "background -- nothing to unload here.)",
                log_file
            )
            log_print("", log_file)
            log_print(f"Log saved to: {log_path}", log_file)


if __name__ == "__main__":
    main()