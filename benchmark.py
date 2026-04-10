#!/usr/bin/env python3
"""
Benchmark inference speed across optimization configurations.

Usage:
    uv run benchmark.py                   # cumulative study
    uv run benchmark.py --ablation        # ablation study
    uv run benchmark.py --ablation --cumulative   # both
    uv run benchmark.py --input path/to/image.jpg --runs 30 --warmup 5
"""

import contextlib
import io
import os
import sys
import time
from pathlib import Path

os.environ["PYOPENGL_PLATFORM"] = ""

import argparse
import cv2
import numpy as np
import torch
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent
VENDOR_SAM3D = REPO_ROOT / "vendor" / "sam-3d-body"
sys.path.insert(0, str(VENDOR_SAM3D))
sys.path.insert(0, str(VENDOR_SAM3D / "notebook"))

try:
    from utils import setup_sam_3d_body
except ImportError as exc:
    logger.error(f"Could not import sam-3d-body utilities: {exc}")
    sys.exit(1)

DEFAULT_MODEL = "facebook/sam-3d-body-dinov3"

# ── Cumulative configs: each row adds one optimization ────────────────────────
CUMULATIVE_CONFIGS = [
    {
        "name":      "baseline (vitdet, full)",
        "detector":  "vitdet",
        "body_only": False,
        "tf32":      False,
        "compile":   False,
    },
    {
        "name":      "+ body-only",
        "detector":  "vitdet",
        "body_only": True,
        "tf32":      False,
        "compile":   False,
    },
    {
        "name":      "+ tf32",
        "detector":  "vitdet",
        "body_only": True,
        "tf32":      True,
        "compile":   False,
    },
    {
        "name":      "+ yolo11n",
        "detector":  "yolo11n.pt",
        "body_only": True,
        "tf32":      True,
        "compile":   False,
    },
    {
        "name":      "+ compile",
        "detector":  "yolo11n.pt",
        "body_only": True,
        "tf32":      True,
        "compile":   True,
    },
]

# ── Ablation configs: start from best, remove one optimization at a time ──────
_BEST = dict(detector="yolo11n.pt", body_only=True, tf32=True, compile=True)

ABLATION_CONFIGS = [
    {**_BEST, "name": "all optimizations (best)"},
    {**_BEST, "name": "w/o compile",   "compile":   False},
    {**_BEST, "name": "w/o tf32",      "tf32":      False},
    {**_BEST, "name": "w/o body-only", "body_only": False},
    {**_BEST, "name": "w/o yolo11n",   "detector":  "vitdet"},
]


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def load_estimator(cfg: dict, model: str) -> object:
    if cfg["tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    is_yolo = cfg["detector"].startswith("yolo")
    with _quiet():
        estimator = setup_sam_3d_body(
            hf_repo_id=model,
            detector_name=None if is_yolo else cfg["detector"],
            segmentor_name="sam2",
            fov_name=None,  # skip MoGe — cam_int is cached after first run anyway
        )
    if is_yolo:
        from yolo_detector import YoloDetector
        estimator.detector = YoloDetector(model_name=cfg["detector"])

    if cfg["compile"]:
        logger.info("  Compiling backbone…")
        estimator.model.backbone = torch.compile(estimator.model.backbone)

    # Warmup pass to trigger CUDA kernel compilation
    dummy = np.zeros((256, 192, 3), dtype=np.uint8)
    with _quiet():
        estimator.process_one_image(dummy)
    torch.cuda.synchronize()

    return estimator


def run_benchmark(estimator, img: np.ndarray, cfg: dict, runs: int, warmup: int):
    inference_type = "body" if cfg["body_only"] else "full"
    timings = []

    total = warmup + runs
    for i in range(total):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with _quiet(), torch.no_grad():
            estimator.process_one_image(img, inference_type=inference_type)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        if i < warmup:
            logger.info(f"  warmup {i+1}/{warmup}  {elapsed*1000:.1f}ms")
        else:
            timings.append(elapsed)

    return np.array(timings) * 1000


def bench_configs(configs, img, model, runs, warmup):
    results = []
    for cfg in configs:
        logger.info(f"{'─'*60}")
        logger.info(f"Config: {cfg['name']}")
        estimator = load_estimator(cfg, model)
        ms = run_benchmark(estimator, img, cfg, runs, warmup)

        mean_ms, std_ms = ms.mean(), ms.std()
        fps = 1000 / mean_ms
        logger.info(f"  {mean_ms:.1f} ± {std_ms:.1f} ms   ({fps:.2f} fps)")
        results.append((cfg["name"], mean_ms, std_ms, fps))

        del estimator
        torch.cuda.empty_cache()

    return results


def print_table(results, baseline_ms, title):
    w = 60
    logger.info(f"\n{'═'*w}")
    logger.info(title)
    logger.info(f"{'─'*w}")
    logger.info(f"{'Config':<35} {'ms':>7}  {'±':>5}  {'fps':>6}  {'vs best':>8}")
    logger.info(f"{'─'*w}")
    for name, mean_ms, std_ms, fps in results:
        delta = mean_ms - baseline_ms
        sign = "+" if delta >= 0 else ""
        logger.info(
            f"{name:<35} {mean_ms:>7.1f}  {std_ms:>5.1f}  {fps:>6.2f}  "
            f"({sign}{delta:.1f} ms)"
        )
    logger.info(f"{'═'*w}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="IMG20260114091943.jpg",
                   help="Image to use for benchmarking")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--runs", type=int, default=20,
                   help="Timed runs per config (after warmup)")
    p.add_argument("--warmup", type=int, default=5,
                   help="Warmup runs to discard")
    p.add_argument("--ablation", action="store_true",
                   help="Run ablation study (remove one optimization at a time from best)")
    p.add_argument("--cumulative", action="store_true",
                   help="Run cumulative study (add one optimization at a time)")
    return p.parse_args()


def main():
    args = parse_args()

    # Default: run cumulative only
    run_cumulative = args.cumulative or not args.ablation
    run_ablation   = args.ablation

    img_path = REPO_ROOT / args.input
    if not img_path.exists():
        logger.error(f"Image not found: {img_path}")
        sys.exit(1)

    img = cv2.imread(str(img_path))
    if img is None:
        logger.error(f"Cannot read image: {img_path}")
        sys.exit(1)
    logger.info(f"Input: {img_path.name}  {img.shape[1]}x{img.shape[0]}")
    logger.info(f"Runs: {args.runs} timed + {args.warmup} warmup each\n")

    if run_cumulative:
        results = bench_configs(CUMULATIVE_CONFIGS, img, args.model, args.runs, args.warmup)
        baseline_ms = results[0][1]
        # Reuse speedup display for cumulative table
        logger.info(f"\n{'═'*60}")
        logger.info("Cumulative optimizations")
        logger.info(f"{'─'*60}")
        logger.info(f"{'Config':<35} {'ms':>7}  {'±':>5}  {'fps':>6}  {'speedup':>8}")
        logger.info(f"{'─'*60}")
        for name, mean_ms, std_ms, fps in results:
            speedup = baseline_ms / mean_ms
            logger.info(f"{name:<35} {mean_ms:>7.1f}  {std_ms:>5.1f}  {fps:>6.2f}  ({speedup:.2f}x)")
        logger.info(f"{'═'*60}")

    if run_ablation:
        results = bench_configs(ABLATION_CONFIGS, img, args.model, args.runs, args.warmup)
        best_ms = results[0][1]
        print_table(results, best_ms, "Ablation study (remove one optimization from best)")


if __name__ == "__main__":
    main()
