#!/usr/bin/env python3
"""
prompt_lab.py - compare vest-question wordings on YOUR frames, per model.

Two steps:

  1. Capture a small labelled test set from the webcam (stop the sentry
     service first; the camera allows only one reader):

         sudo systemctl stop icp-safety-sentry
         python3 prompt_lab.py capture --out data/prompt_lab

     You are asked for a label before each shot, then get 3 seconds to pose:
         P1V1            one person wearing a vest
         P1V0            one person NOT wearing one
         P1V0 held       add a word after the label as a note (here:
                         vest held in the hand, which must not count)
         P2V1            two people, one vested
         P0V0            empty scene
     Take at least 3 of P1V1 and 3 of P1V0 (a few with the vest held or
     draped). A test set with only vested people cannot tell a good
     prompt from one that always says GO.

  2. Score every candidate wording on every model:

         python3 prompt_lab.py run --frames data/prompt_lab

     Prints, per model and candidate: verdict accuracy (GO / STOP /
     NO PERSON against your labels), people-count and vest-count accuracy,
     failures (unparseable answers or Ollama errors), and average seconds
     per frame. Full per-frame detail is saved to a results file.

Candidates A to C are wordings already used in this project. D and E are
NEW and untested: D is a shorter combined question, and E asks the same
thing but makes Ollama enforce a JSON answer (structured output), which
guarantees a parseable reply. Whether either beats A to C is exactly
what this run measures.

Limits: it scores the counting questions only. The torso description used
for the advisory sentence does not affect the verdict, so it is skipped.
Each model is loaded alone and unloaded afterwards, so run it with the
sentry service stopped and nothing else holding the GPU.

Progress shows live, one mark per frame (. = verdict correct, X = wrong,
E = Ollama error). Results are saved to the results file as they arrive,
and Ctrl+C prints a partial table instead of discarding the run.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import ollama

# name -> (ollama tag, num_ctx). Same context sizes as the detector classes.
MODELS = {
    "ministral": ("ministral-3:3b", 4096),
    "cosmos": ("hf.co/apolo13x/Cosmos-Reason2-2B-GGUF:Q4_K_M", 8192),
    "gemma3": ("gemma3:4b", 4096),
}

BREVITY = (
    "Reply with only the answer: a bare number for counting questions, or "
    "a few words for descriptions. No sentences and no explanation."
)

LONG_VEST = (
    "How many people in this image are properly wearing a high-visibility "
    "orange, yellow, or lime green safety vest on their torso?\n\n"
    "Do NOT count:\n"
    "- a vest being held in a hand\n"
    "- a vest being carried\n"
    "- a vest resting on a shoulder\n"
    "- a vest partway through being put on\n"
    "- a vest lying or hanging nearby but not worn\n"
    "- ordinary clothing that is not a safety vest\n\n"
    "Answer with just a number."
)

COMBINED_LONG = """
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

SHORT_RULE = (
    "Count the people in this image, and how many of them are wearing a "
    "high-visibility safety vest on their body. A vest held in the hand "
    "or lying nearby does not count."
)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "people": {"type": "integer"},
        "vests": {"type": "integer"},
    },
    "required": ["people", "vests"],
}

CANDIDATES = {
    # PaliGemma's questions word for word (the v4 detector), two calls.
    "A_paligemma_split": dict(
        kind="split",
        system=BREVITY,
        q_people="answer en How many people are visible in this image?",
        q_vests=(
            "answer en How many people are wearing a high-visibility safety "
            "vest on their body, not just holding one?"
        ),
    ),
    # The two-call prompts that scored 100% in the benchmark suite.
    "B_benchmark_split": dict(
        kind="split",
        system=None,
        q_people="How many people are visible in this image? Answer with just a number.",
        q_vests=LONG_VEST,
    ),
    # The single combined prompt that scored 100% in the benchmark suite.
    "C_combined_long": dict(kind="combined", system=None, prompt=COMBINED_LONG, fmt=None),
    # NEW: shorter combined question, labelled two-line reply.
    "D_combined_short": dict(
        kind="combined",
        system=None,
        prompt=SHORT_RULE
        + "\n\nReply in exactly this format and nothing else:\n"
        "PEOPLE: <number>\nVESTS: <number>",
        fmt=None,
    ),
    # NEW: same short question, answer forced into a JSON schema by Ollama.
    "E_combined_json": dict(kind="combined", system=None, prompt=SHORT_RULE, fmt=JSON_SCHEMA),
}

_WORD_NUMBERS = {
    "zero": 0, "no": 0, "none": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

LABEL_RE = re.compile(r"__P(\d+)V(\d+)")


# ---------------------------------------------------------------------
# Parsing and scoring
# ---------------------------------------------------------------------

def parse_int(text: str) -> int | None:
    """First number in the answer (digits, else a number word); None if there
    is none. None is a parse failure, which is different from an answer of 0."""
    match = re.search(r"\d+", text)
    if match:
        return int(match.group(0))
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in _WORD_NUMBERS:
            return _WORD_NUMBERS[word]
    return None


def verdict(people: int, vests: int) -> str:
    if people == 0:
        return "NO PERSON"
    return "GO" if min(vests, people) >= people else "STOP"


def expected_from_name(name: str) -> tuple[int, int] | None:
    match = LABEL_RE.search(name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


# ---------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------

def ask(model, num_ctx, jpeg, text, system=None, fmt=None, max_tokens=32):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": text, "images": [jpeg]})
    kwargs = dict(
        model=model,
        messages=messages,
        options={"temperature": 0.0, "num_ctx": num_ctx, "num_predict": max_tokens},
    )
    if fmt is not None:
        kwargs["format"] = fmt
    start = time.perf_counter()
    response = ollama.chat(**kwargs)
    return response["message"]["content"].strip(), time.perf_counter() - start


def run_candidate(cand, model, num_ctx, jpeg) -> dict:
    """Returns {people, vests, seconds, raw}. people/vests are None when the
    answer could not be parsed."""
    if cand["kind"] == "split":
        raw1, t1 = ask(model, num_ctx, jpeg, cand["q_people"], cand["system"])
        people = parse_int(raw1)
        if people is None:
            return dict(people=None, vests=None, seconds=t1, raw=[raw1])
        if people == 0:
            return dict(people=0, vests=0, seconds=t1, raw=[raw1])
        raw2, t2 = ask(model, num_ctx, jpeg, cand["q_vests"], cand["system"])
        return dict(people=people, vests=parse_int(raw2), seconds=t1 + t2, raw=[raw1, raw2])

    raw, t = ask(model, num_ctx, jpeg, cand["prompt"], cand["system"], cand["fmt"])
    people = vests = None
    if cand["fmt"] is not None:
        try:
            data = json.loads(raw)
            people, vests = int(data["people"]), int(data["vests"])
        except Exception:
            pass
    else:
        mp = re.search(r"PEOPLE\s*:\s*(\d+)", raw, re.IGNORECASE)
        mv = re.search(r"VESTS\s*:\s*(\d+)", raw, re.IGNORECASE)
        if mp and mv:
            people, vests = int(mp.group(1)), int(mv.group(1))
    return dict(people=people, vests=vests, seconds=t, raw=[raw])


# ---------------------------------------------------------------------
# run
# ---------------------------------------------------------------------

def cmd_run(args) -> None:
    frames_dir = Path(args.frames).expanduser()
    frames = []
    for path in sorted(frames_dir.glob("*.jp*g")) + sorted(frames_dir.glob("*.png")):
        exp = expected_from_name(path.name)
        if exp is None:
            print(f"skipping {path.name}: no __P<people>V<vests> label in the name")
            continue
        frames.append((path, exp, path.read_bytes()))
    if not frames:
        raise SystemExit(f"No labelled frames in {frames_dir}. Run the capture step first.")

    n_vested = sum(1 for _, (p, v), _ in frames if p > 0 and v >= p)
    n_unvested = sum(1 for _, (p, v), _ in frames if p > 0 and v < p)
    print(f"{len(frames)} frames: {n_vested} compliant, {n_unvested} non-compliant, "
          f"{len(frames) - n_vested - n_unvested} empty")
    if n_unvested == 0:
        print("WARNING: no non-compliant frames. This cannot tell a good prompt from one "
              "that always says GO. Capture some P1V0 frames.")

    wanted_models = [m.strip() for m in args.models.split(",")]
    wanted_cands = [c for c in CANDIDATES if not args.candidates
                    or any(c.startswith(x.strip()) for x in args.candidates.split(","))]

    pulled = {m["model"] for m in ollama.list().get("models", [])}

    out = frames_dir / f"prompt_lab_results_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
    fh = out.open("w", encoding="utf-8")
    print(f"Results are written to {out} as they arrive, so a stopped run keeps them.")

    def save(line: str) -> None:
        fh.write(line + "\n")
        fh.flush()

    header = (f"{'model':10s} {'candidate':20s} {'verdict':>8s} {'people':>7s} "
              f"{'vests':>6s} {'fails':>6s} {'sec/frame':>10s}")

    def row_text(row: tuple) -> str:
        short, cname, sv, sp, sc, fl, sec, dead = row
        if dead:
            return f"{short:10s} {cname:20s} {'ERR':>8s} {'ERR':>7s} {'ERR':>6s} {fl:6d} {'-':>10s}"
        return f"{short:10s} {cname:20s} {sv:8.0%} {sp:7.0%} {sc:6.0%} {fl:6d} {sec:10.2f}"

    summary: list[tuple] = []
    current_tag = None
    interrupted = False
    save(header)

    try:
        for short in wanted_models:
            if short not in MODELS:
                print(f"unknown model '{short}', choose from {list(MODELS)}")
                continue
            tag, num_ctx = MODELS[short]
            if tag not in pulled:
                print(f"{short}: '{tag}' is not pulled, skipping")
                continue

            print(f"\n=== {short} ({tag}) ===")
            current_tag = tag
            try:  # absorb the model-load time before anything is timed
                ask(tag, num_ctx, frames[0][2], "How many people are visible in this image?")
            except Exception as e:
                print(f"  could not warm up {short}: {str(e)[:100]}; skipping this model")
                continue

            for cname in wanted_cands:
                cand = CANDIDATES[cname]
                n = ok_v = ok_p = ok_s = fails = 0
                secs = 0.0
                consecutive_errors = 0
                abandoned = False
                misses: list[str] = []
                print(f"  {cname:19s} ", end="", flush=True)  # one mark per frame: . ok, X miss
                for path, (ep, ev), jpeg in frames:
                    try:
                        res = run_candidate(cand, tag, num_ctx, jpeg)
                        consecutive_errors = 0
                    except Exception as e:
                        consecutive_errors += 1
                        fails += 1
                        n += 1
                        line = f"{short:9s} {cname:19s} {path.name:34s} ERROR {str(e)[:90]}"
                        save(line)
                        misses.append(line)
                        print("E", end="", flush=True)
                        if consecutive_errors >= 3:
                            abandoned = True
                            break
                        time.sleep(2)
                        continue

                    n += 1
                    secs += res["seconds"]
                    if res["people"] is None or res["vests"] is None:
                        fails += 1
                        line_ok = False
                        got = "unparseable"
                    else:
                        p, v = res["people"], min(res["vests"], res["people"])
                        ok_p += p == ep
                        ok_v += v == min(ev, ep)
                        line_ok = verdict(p, v) == verdict(ep, ev)
                        ok_s += line_ok
                        got = f"P{p}V{v} {verdict(p, v)}"
                    line = (f"{short:9s} {cname:19s} {path.name:34s} want P{ep}V{ev} "
                            f"{verdict(ep, ev):9s} got {got:18s} {res['seconds']:5.2f}s "
                            f"{'ok ' if line_ok else 'MISS'} raw={res['raw']!r}")
                    save(line)  # written immediately, not at the end
                    print("." if line_ok else "X", end="", flush=True)
                    if args.verbose or not line_ok:
                        misses.append(line)

                if n:
                    timed = max(n - fails, 1)
                    row = (short, cname, ok_s / n, ok_p / n, ok_v / n, fails,
                           secs / timed, abandoned)
                    summary.append(row)
                    save("SUMMARY " + row_text(row))
                    if abandoned:
                        print("  abandoned after 3 errors in a row")
                    else:
                        print(f"  verdict {ok_s / n:.0%}, {secs / timed:.2f}s/frame, {fails} failed")
                    for m in misses:
                        print("      " + m)

            try:  # free the model before loading the next one
                ollama.chat(model=tag, messages=[], keep_alive=0)
            except Exception:
                pass
            current_tag = None
    except KeyboardInterrupt:
        interrupted = True
        print("\n\nInterrupted. Showing what completed before the stop.")
        if current_tag:
            try:
                ollama.chat(model=current_tag, messages=[], keep_alive=0)
            except Exception:
                pass

    table = [header, "-" * len(header)] + [row_text(r) for r in summary]
    text = "\n".join(table)
    print("\n" + text)
    if interrupted:
        print("(partial: only the candidates listed above finished)")
    save("\nFINAL TABLE" + (" (PARTIAL, run was interrupted)" if interrupted else ""))
    save(text)
    fh.close()
    print(f"\nPer-frame detail saved to {out}")


# ---------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------

def default_source() -> str:
    paths = sorted(glob.glob("/dev/v4l/by-id/*video-index0"))
    return paths[0] if paths else "0"


def cmd_capture(args) -> None:
    import cv2  # only needed here

    # Check the folder first, so a permissions problem fails before the camera is touched.
    out = Path(args.out).expanduser()
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".write_test"
        probe.write_text("x")
        probe.unlink()
    except OSError as e:
        raise SystemExit(
            f"Cannot write to '{out}': {e}\n"
            "Pick a folder you own, e.g. --out ~/prompt_lab_frames "
            "(outside the git repo is better: these are photos of people)."
        )

    source = args.source or default_source()
    # CAP_V4L2 skips OpenCV's GStreamer attempt, which prints warnings and
    # fails when the device is busy.
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source, cv2.CAP_V4L2)
    ok, _ = cap.read() if cap.isOpened() else (False, None)
    if not ok:
        cap.release()
        raise SystemExit(
            f"Could not read a frame from camera '{source}'. The usual cause is the "
            "sentry service still holding it: sudo systemctl stop icp-safety-sentry"
        )
    n = len(list(out.glob("cap*__P*.jpg")))
    print(f"Camera: {source}\nSaving to: {out}\n")
    print("Labels: P1V1 = one person with a vest, P1V0 = one person without one,\n"
          "        P2V1 = two people one vest, P0V0 = empty. Add a note after a space,\n"
          "        e.g. 'P1V0 held'. Press Enter on an empty line to finish.\n")
    try:
        while True:
            s = input("Label: ").strip()
            if not s:
                break
            m = re.match(r"^P(\d+)V(\d+)(?:\s+(\w+))?$", s, re.IGNORECASE)
            if not m:
                print("  use the form P1V1, P1V0, P2V1 or P0V0 (optional note after a space)")
                continue
            people, vests, note = int(m.group(1)), int(m.group(2)), m.group(3)
            if vests > people:
                print("  vests cannot exceed people")
                continue
            for i in (3, 2, 1):
                print(f"  {i}...")
                time.sleep(1)
            for _ in range(5):  # drain stale buffered frames
                cap.read()
            ok, frame = cap.read()
            if not ok:
                print("  could not read a frame")
                continue
            name = f"cap{n:03d}__P{people}V{vests}" + (f"_{note}" if note else "") + ".jpg"
            cv2.imwrite(str(out / name), frame)
            n += 1
            print(f"  saved {name}")
    finally:
        cap.release()


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)  # live output even through `| tee`
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="take labelled frames from the webcam")
    c.add_argument("--source", help="camera index or device path (default: the first /dev/v4l/by-id/*video-index0)")
    c.add_argument("--out", default="data/prompt_lab")
    c.set_defaults(fn=cmd_capture)

    r = sub.add_parser("run", help="score every candidate on every model")
    r.add_argument("--frames", default="data/prompt_lab")
    r.add_argument("--models", default=",".join(MODELS), help="comma list from: " + ", ".join(MODELS))
    r.add_argument("--candidates", default="", help="comma list of name prefixes, e.g. A,C,E (default: all)")
    r.add_argument("--verbose", action="store_true", help="print every frame, not only misses")
    r.set_defaults(fn=cmd_run)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
