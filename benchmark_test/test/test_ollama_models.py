import re
import time
from pathlib import Path
from datetime import datetime

import ollama

# ============================================================
# Model registry
#
# Add new models here as you pull more -- (display_name, ollama
# model_name, log_filename_slug). Nothing else in this script
# needs to change.
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
            "num_ctx": 8192,  # default 4096 too small once image tokens are added
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
    print("  [1] Quick single-image test (console only, ~2 calls)")
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

    people_answer, people_time = ask(model_name, IMAGE_PATH, PEOPLE_PROMPT)
    print(f"People prompt -> '{people_answer}'  ({people_time:.3f}s)")

    vest_answer, vest_time = ask(model_name, IMAGE_PATH, VEST_PROMPT)
    print(f"Vest prompt   -> '{vest_answer}'  ({vest_time:.3f}s)")

    print("-" * 60)
    print(f"Total: {people_time + vest_time:.3f}s")


# ============================================================
# Full 200-inference benchmark (logged, warmup excluded)
# ============================================================

def run_benchmark(display_name: str, model_name: str, slug: str):

    MAX_INFERENCES = 200

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = LOG_DIR / f"{timestamp}_{slug}_ollama_image.txt"

    with open(log_path, "w", encoding="utf-8") as log_file:

        log_print("=" * 60, log_file)
        log_print(
            f"{display_name.upper()} (OLLAMA) + STATIC IMAGE "
            f"REPEATED-INFERENCE TEST",
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

        _ = ask(model_name, IMAGE_PATH, PEOPLE_PROMPT)

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

                people_answer, people_time = ask(model_name, IMAGE_PATH, PEOPLE_PROMPT)
                n_people = parse_number(people_answer)

                # -------------------------------------------------
                # 2. Count vests
                # -------------------------------------------------

                vest_answer, vest_time = ask(model_name, IMAGE_PATH, VEST_PROMPT)
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