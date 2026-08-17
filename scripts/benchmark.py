"""Benchmark detector backends on a folder of sample images.

Usage:
    uv run --extra yolo --extra vlm python scripts/benchmark.py --images data/samples --backend yolo_hsv
    uv run --extra yolo --extra vlm python scripts/benchmark.py --images data/samples --backend vlm_qwen2vl
    uv run --extra yolo --extra vlm python scripts/benchmark.py --images data/samples --backend vlm_moondream

Reports per-image latency and verdict; prints summary stats (mean/p50/p95 ms).
Ground truth: name files like `<anything>__GO.jpg` / `<anything>__STOP.jpg` to
also get accuracy against the filename label.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import cv2

from icp_safety_vlm.detectors import get_detector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()

    detector = get_detector(args.backend)
    detector.warmup()

    latencies = []
    correct = 0
    labeled = 0

    for path in sorted(args.images.glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        result = detector.timed_infer(frame)
        latencies.append(result.latency_ms)

        label = None
        stem = path.stem.upper()
        if stem.endswith("__GO"):
            label = "GO"
        elif stem.endswith("__STOP"):
            label = "STOP"
        if label:
            labeled += 1
            correct += int(result.status == label)

        print(f"{path.name:40s} status={result.status:5s} n_people={len(result.detections):2d} latency={result.latency_ms:7.1f}ms")

    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
        print(f"\nbackend={args.backend} n={len(latencies)} mean={statistics.mean(latencies):.1f}ms p50={p50:.1f}ms p95={p95:.1f}ms")
    if labeled:
        print(f"accuracy={correct}/{labeled} ({100*correct/labeled:.1f}%)")


if __name__ == "__main__":
    main()
