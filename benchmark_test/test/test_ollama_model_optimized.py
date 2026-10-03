import re
import time
from pathlib import Path
from datetime import datetime

import ollama

# ============================================================
# Model registry
#
# Same models as test_ollama_models.py (the two-call baseline
# selector) -- this script tests the same models with the
# single-call "optimized" prompt architecture instead, for a
# direct baseline-vs-optimized comparison per model.
# ============================================================

MODELS = {
    "1": ("Gemma 3 4B", "gemma3:4b", "gemma3_4b"),
    "2": ("Ministral 3 3B Instruct", "ministral-3:3b", "ministral3_3b"),
    "3": ("Moondream", "moondream", "moondream"),
    "4": ("LLaVA-Phi3", "llava-phi3", "llava_phi3"),
    "5": (
        "Cosmos Reason 2 2B (community GGUF)",
        "hf.co/apolo13x/Cosmos-Reason2-2B-GGUF:Q4_K_M",
        "cosmos_reason2_2b",
    ),
    "6": ("LLaVA-Llama3", "llava-llama3", "llava_llama3"),
    "7": ("Granite Vision", "granite3.2-vision", "granite_vision"),
    "8": ("BakLLaVA", "bakllava", "bakllava"),
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = PROJECT_ROOT / "images" / "test.jpg"

LOG_DIR = PROJECT_ROOT / "logs"

MAX_INFERENCES = 200

# ============================================================
# Optimized VLM Prompt
#
# Single combined call, same wording as the RTX optimized
# detector (RTXQwenVLOptimizedDetector) -- one call asks for
# both counts at once, instead of two separate calls.
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


# ============================================================
# Logging
# ============================================================

def log_print(message, log_file=None):
    """Print message to terminal and, if a log file is open, save it there too."""
    print(message)

    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


# ============================================================
# Model helper
# ============================================================

def ask(model_name: str, image_path: Path, prompt: str) -> tuple[str, float]:
    """Send one image + text prompt to the model. Returns (answer, seconds)."""

    start = time.perf_counter()

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],
            }
        ],
        options={
            "temperature": 0.0,  # deterministic, matches every other detector in this project
            "num_ctx": 8192,  # default 4096 is too small once image tokens are added for some models
        },
    )

    elapsed = time.perf_counter() - start

    answer = response["message"]["content"].strip()

    return answer, elapsed


def parse_count(text: str, label: str) -> int:
    """
    Extract an integer from responses such as:

        PEOPLE: 2
        VESTS: 1

    Returns 0 if the requested label cannot be parsed -- same fallback
    behavior as every other parser in this project, and matches
    RTXQwenVLOptimizedDetector's parse_count().
    """

    if not text:
        return 0

    pattern = rf"{label}\s*:\s*(\d+)"

    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return int(match.group(1))

    return 0


# ============================================================
# Selection prompts
# ============================================================

def select_model() -> tuple[str, str, str]:
    print("Select a model:")
    for key, (display_name, _, _) in MODELS.items():
        print(f"  [{key}] {display_name}")

    choice = input("Enter number: ").strip()

    if choice not in MODELS:
        raise ValueError(f"'{choice}' is not a valid option.")

    return MODELS[choice]  # (display_name, model_name, slug)


def select_mode() -> str:
    print("\nSelect test mode:")
    print("  [1] Quick single-image test (console only, 1 call)")
    print("  [2] Full 200-inference benchmark (logged to file)")

    choice = input("Enter number: ").strip()

    if choice not in ("1", "2"):
        raise ValueError(f"'{choice}' is not a valid option.")

    return choice


# ============================================================
# Quick single-image test (no log file, no warmup, no loop)
# ============================================================

def run_quick_test(display_name: str, model_name: str):

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}\n"
            f"Update IMAGE_PATH at the top of this script."
        )

    print(f"\nModel: {display_name} ({model_name})")
    print(f"Image: {IMAGE_PATH}")
    print("-" * 60)

    response, elapsed = ask(model_name, IMAGE_PATH, SAFETY_PROMPT)

    people = parse_count(response, "PEOPLE")
    vests = min(parse_count(response, "VESTS"), people)

    print(f"Raw response:\n{response}")
    print("-" * 60)
    print(f"Parsed -> People: {people}  Vests: {vests}  ({elapsed:.3f}s)")


# ============================================================
# Full 200-inference benchmark (logged, warmup excluded)
# ============================================================

def run_benchmark(display_name: str, model_name: str, slug: str):

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = LOG_DIR / f"{timestamp}_{slug}_ollama_optimized_image.txt"

    with open(log_path, "w", encoding="utf-8") as log_file:

        log_print("=" * 60, log_file)
        log_print(
            f"{display_name.upper()} (OLLAMA, SINGLE-CALL OPTIMIZED "
            f"PROMPT) + STATIC IMAGE REPEATED-INFERENCE TEST",
            log_file
        )
        log_print("=" * 60, log_file)

        log_print(f"Log file: {log_path}", log_file)
        log_print(f"Model: {model_name}", log_file)
        log_print(f"Image path: {IMAGE_PATH}", log_file)

        if not IMAGE_PATH.is_file():
            raise FileNotFoundError(
                f"Image not found:\n{IMAGE_PATH}\n"
                f"Update IMAGE_PATH at the top of this script, or "
                f"place a test image at that path."
            )

        # ----------------------------------------------------
        # Throwaway warmup inference
        # ----------------------------------------------------

        log_print(
            "Running warmup inference "
            "(absorbs first-call model-load cost)...",
            log_file
        )

        warmup_start = time.perf_counter()

        _ = ask(model_name, IMAGE_PATH, SAFETY_PROMPT)

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
                # ONE combined call
                # -------------------------------------------------

                response, inference_time = ask(model_name, IMAGE_PATH, SAFETY_PROMPT)

                n_people = parse_count(response, "PEOPLE")
                n_vests = min(parse_count(response, "VESTS"), n_people)

                total_time += inference_time

                # -------------------------------------------------
                # Result
                # -------------------------------------------------

                status = "GO" if (n_people > 0 and n_vests >= n_people) else (
                    "NO_PERSON" if n_people == 0 else "STOP"
                )

                log_print("", log_file)
                log_print("-" * 60, log_file)

                log_print(f"Raw response: {response!r}", log_file)
                log_print(f"People: {n_people}", log_file)
                log_print(f"Vests: {n_vests}", log_file)
                log_print(f"Status: {status}", log_file)

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

                if frame_count >= MAX_INFERENCES:

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


# ============================================================
# Main
# ============================================================

def main():

    display_name, model_name, slug = select_model()

    mode = select_mode()

    if mode == "1":
        run_quick_test(display_name, model_name)
    else:
        run_benchmark(display_name, model_name, slug)


if __name__ == "__main__":
    main()
