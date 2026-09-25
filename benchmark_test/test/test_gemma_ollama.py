import time
from pathlib import Path

import ollama

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "gemma3:4b"

IMAGE_PATH = PROJECT_ROOT / "images" / "test.jpg"

# Same count-based prompt shape used throughout this project --
# direct yes/no or GO/STOP classification prompts are known to
# trigger acquiescence bias on small VLMs (see project notes).
# Asking for counts instead has proven reliable across every
# other model tested so far.
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


def main():

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}\n"
            f"Update IMAGE_PATH at the top of this script."
        )

    print(f"Model: {MODEL_NAME}")
    print(f"Image: {IMAGE_PATH}")
    print("-" * 60)

    people_answer, people_time = ask(IMAGE_PATH, PEOPLE_PROMPT)
    print(f"People prompt -> '{people_answer}'  ({people_time:.3f}s)")

    vest_answer, vest_time = ask(IMAGE_PATH, VEST_PROMPT)
    print(f"Vest prompt   -> '{vest_answer}'  ({vest_time:.3f}s)")

    print("-" * 60)
    print(f"Total: {people_time + vest_time:.3f}s")


if __name__ == "__main__":
    main()