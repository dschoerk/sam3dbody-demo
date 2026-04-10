#!/usr/bin/env python3
"""
SAM-3D-Body Demo — 3D human mesh recovery for images and video.

Usage:
    uv run demo.py --input image.jpg
    uv run demo.py --input image.jpg --show --save-meshes
    uv run demo.py --input video.mp4 --output result.mp4
    uv run demo.py --input video.mp4 --frame-skip 2

Prerequisites:
    Run ./setup.sh once to clone the model repo and authenticate with HuggingFace.
"""

import os
import sys
# Force UTF-8 I/O so ✓/✗ characters don't crash on Windows CP1252 consoles.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
# Must be set BEFORE any pyrender/OpenGL import.
# Empty string = use WGL (Windows native) so the import succeeds.
# pyrender's OffscreenRenderer still won't work without osmesa/EGL,
# but we catch that and fall back to a matplotlib 3D joint plot.
os.environ["PYOPENGL_PLATFORM"] = ""

import argparse
import collections
import contextlib
import queue
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
VENDOR_SAM3D = REPO_ROOT / "vendor" / "sam-3d-body"

if not VENDOR_SAM3D.exists():
    logger.error(
        f"sam-3d-body not found at {VENDOR_SAM3D}.\n"
        "Please run:  bash setup.sh"
    )
    sys.exit(1)

# Insert both the repo root (for sam_3d_body package) and notebook/ (for utils)
sys.path.insert(0, str(VENDOR_SAM3D))
sys.path.insert(0, str(VENDOR_SAM3D / "notebook"))

# pyrender needs a display backend on Windows — use pyglet (default) or set
# PYOPENGL_PLATFORM=osmesa for headless.  Leave unset for normal desktop use.
# os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

# ── Imports from vendored sam-3d-body ─────────────────────────────────────────
try:
    from utils import (  # noqa: E402  (notebook/utils.py)
        setup_sam_3d_body,
        setup_visualizer,
        visualize_2d_results,
        visualize_3d_mesh,
        save_mesh_results,
    )
except ImportError as exc:
    logger.error(f"Could not import sam-3d-body utilities: {exc}")
    logger.error("Make sure setup.sh completed successfully.")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "facebook/sam-3d-body-dinov3"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _autocast_ctx(args):
    """Return an autocast context if --bf16 was passed, otherwise a no-op."""
    if args.bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@contextlib.contextmanager
def _quiet():
    """Suppress stdout from vendor code (debug prints inside process_one_image)."""
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAM-3D-Body: 3D human mesh recovery for images and video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", default=None, help="Image or video file path")
    p.add_argument(
        "--output", "-o", default=None,
        help="Output path (default: ./output/<input_name>)",
    )
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="HuggingFace repo ID or local checkpoint path",
    )
    p.add_argument(
        "--no-detector", action="store_true",
        help="Skip automatic person detector (treat whole image as one person)",
    )
    p.add_argument(
        "--no-fov", action="store_true",
        help="Skip MoGe FOV estimation (faster, less accurate depth scale)",
    )
    p.add_argument(
        "--bbox-thresh", type=float, default=0.6,
        help="Person detection confidence threshold",
    )
    p.add_argument(
        "--save-meshes", action="store_true",
        help="Save per-person 3D meshes as .ply files alongside the output",
    )
    p.add_argument(
        "--frame-skip", type=int, default=1,
        help="[Video] Process every Nth frame (1 = every frame)",
    )
    p.add_argument(
        "--show", action="store_true",
        help="[Image] Open an interactive matplotlib window after processing",
    )
    p.add_argument(
        "--webcam", action="store_true",
        help="[Webcam] Run real-time inference from webcam (no --input needed)",
    )
    p.add_argument(
        "--webcam-id", type=int, default=0,
        help="[Webcam] Camera device index (default: 0)",
    )
    p.add_argument(
        "--no-3d", action="store_true",
        help="[Webcam] Hide the 3D skeleton panel (show only the 2D overlay)",
    )
    p.add_argument(
        "--body-only", action="store_true",
        help="Run body decoder only — skips the two hand forward passes (~3× faster, less accurate fingers)",
    )
    p.add_argument(
        "--bf16", action="store_true",
        help="Run inference under torch.autocast bfloat16 (faster on Ampere+, RTX 30/40 series)",
    )
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────
def load_estimator(args: argparse.Namespace):
    logger.info(f"Loading SAM-3D-Body model: {args.model}")
    estimator = setup_sam_3d_body(
        hf_repo_id=args.model,
        detector_name=None if args.no_detector else "vitdet",
        segmentor_name="sam2",
        fov_name=None if args.no_fov else "moge2",
    )
    logger.info("Model ready.")
    return estimator


# ── Image processing ──────────────────────────────────────────────────────────
def run_image(estimator, image_path: Path, output_path: Path, args: argparse.Namespace):
    logger.info(f"Processing image: {image_path}")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        logger.error(f"Cannot read image: {image_path}")
        return

    with _quiet(), _autocast_ctx(args):
        outputs = estimator.process_one_image(str(image_path))
    if not outputs:
        logger.warning("No humans detected.")
        return

    logger.info(f"Detected {len(outputs)} person(s).")
    visualizer = setup_visualizer()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 2D skeleton overlay ──
    vis_2d = visualize_2d_results(img_bgr, outputs, visualizer)
    overlay_path = output_path.with_suffix("").with_suffix(".overlay.jpg")
    if vis_2d:
        combined = np.hstack(vis_2d) if len(vis_2d) > 1 else vis_2d[0]
        cv2.imwrite(str(overlay_path), combined)
        logger.info(f"Saved 2D overlay → {overlay_path}")

    # ── 3D mesh multi-view (requires osmesa/EGL; may not work on Windows) ──
    try:
        mesh_views = visualize_3d_mesh(img_bgr, outputs, estimator.faces)
        if mesh_views:
            for i, view in enumerate(mesh_views):
                view_path = output_path.with_suffix("").with_suffix(f".3d_p{i}.jpg")
                cv2.imwrite(str(view_path), view)
            logger.info(f"Saved {len(mesh_views)} 3D view(s) alongside overlay.")
    except Exception as e:
        logger.warning(f"pyrender offscreen not available ({e}); using matplotlib 3D fallback.")

    # ── Matplotlib 3D skeleton fallback (always works, no osmesa needed) ──
    skel_3d_path = output_path.with_suffix("").with_suffix(".3d_skeleton.png")
    _save_matplotlib_3d(outputs, skel_3d_path)

    # ── .ply meshes (optional) ──
    if args.save_meshes:
        try:
            mesh_dir = output_path.parent / "meshes" / image_path.stem
            save_mesh_results(img_bgr, outputs, estimator.faces, str(mesh_dir), image_path.stem)
            logger.info(f"Saved .ply meshes → {mesh_dir}")
        except Exception as e:
            logger.warning(f".ply save skipped: {e}")

    # ── Interactive display ──
    if args.show:
        fig, axes = plt.subplots(1, 2 if vis_2d else 1, figsize=(14, 7))
        if vis_2d:
            ax_img = axes[0] if hasattr(axes, '__len__') else axes
            combined = np.hstack(vis_2d) if len(vis_2d) > 1 else vis_2d[0]
            ax_img.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
            ax_img.set_title("2D Skeleton Overlay")
            ax_img.axis("off")
        skel_img = plt.imread(str(skel_3d_path))
        ax_3d = axes[1] if hasattr(axes, '__len__') and len(axes) > 1 else axes
        ax_3d.imshow(skel_img)
        ax_3d.set_title("3D Skeleton")
        ax_3d.axis("off")
        plt.tight_layout()
        plt.show()


def _save_matplotlib_3d(outputs: list, save_path: Path):
    """Render the predicted 3D skeleton joints using matplotlib and save as PNG."""
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10.colors
    for person_idx, person in enumerate(outputs):
        kp3d = person.get("pred_keypoints_3d")
        if kp3d is None:
            continue
        # kp3d: (N_joints, 3) in metres, camera coords (Y-down, Z-forward)
        if hasattr(kp3d, "cpu"):
            kp3d = kp3d.cpu().numpy()
        kp3d = np.array(kp3d)
        if kp3d.ndim == 2 and kp3d.shape[1] >= 3:
            xs, ys, zs = kp3d[:, 0], kp3d[:, 1], kp3d[:, 2]
            col = colors[person_idx % len(colors)]
            ax.scatter(xs, ys, zs, c=[col], s=20, depthshade=True, label=f"Person {person_idx}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (depth)")
    ax.set_title("3D Joint Positions")
    ax.view_init(elev=-80, azim=-90)  # front-facing view
    if outputs:
        ax.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    logger.info(f"Saved 3D skeleton plot → {save_path}")


# ── Video processing ──────────────────────────────────────────────────────────
def run_video(estimator, video_path: Path, output_path: Path, args: argparse.Namespace):
    logger.info(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    effective_fps = fps / max(args.frame_skip, 1)

    logger.info(f"  {width}×{height}  {fps:.2f} fps  {total} frames total")
    logger.info(f"  frame-skip={args.frame_skip}  output fps≈{effective_fps:.2f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, effective_fps, (width, height))

    visualizer = setup_visualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_idx = 0
        written = 0
        pbar = tqdm(total=total, desc="Frames", unit="fr")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % args.frame_skip == 0:
                tmp_path = os.path.join(tmpdir, f"frame_{frame_idx:07d}.jpg")
                cv2.imwrite(tmp_path, frame)

                try:
                    with _quiet(), _autocast_ctx(args):
                        outputs = estimator.process_one_image(tmp_path)
                except Exception as exc:
                    logger.warning(f"Frame {frame_idx}: inference error — {exc}")
                    outputs = None

                if outputs:
                    vis_list = visualize_2d_results(frame, outputs, visualizer)
                    out_frame = vis_list[0] if vis_list else frame
                else:
                    out_frame = frame

                writer.write(out_frame)
                written += 1

            frame_idx += 1
            pbar.update(1)

        pbar.close()

    cap.release()
    writer.release()
    logger.info(f"Saved video ({written} frames) → {output_path}")


# ── Webcam 3D rendering ───────────────────────────────────────────────────────
# Bone pairs derived directly from mhr70.py skeleton_info (name → index via keypoint_info).
# Each entry: (joint_a, joint_b, bgr_color)
# Colors are the official RGB values from skeleton_info converted to BGR for OpenCV.
_BONES = [
    # ── legs (left=green, right=orange) ──────────────────────────────────────
    (13, 11, (  0, 255,   0)),  # left_ankle  → left_knee
    (11,  9, (  0, 255,   0)),  # left_knee   → left_hip
    (14, 12, (  0, 128, 255)),  # right_ankle → right_knee
    (12, 10, (  0, 128, 255)),  # right_knee  → right_hip
    # ── torso / hips (blue) ──────────────────────────────────────────────────
    ( 9, 10, (255, 153,  51)),  # left_hip    → right_hip
    ( 5,  9, (255, 153,  51)),  # left_shoulder  → left_hip
    ( 6, 10, (255, 153,  51)),  # right_shoulder → right_hip
    ( 5,  6, (255, 153,  51)),  # left_shoulder  → right_shoulder
    # ── arms ─────────────────────────────────────────────────────────────────
    ( 5,  7, (  0, 255,   0)),  # left_shoulder  → left_elbow
    ( 6,  8, (  0, 128, 255)),  # right_shoulder → right_elbow
    ( 7, 62, (  0, 255,   0)),  # left_elbow  → left_wrist  (62)
    ( 8, 41, (  0, 128, 255)),  # right_elbow → right_wrist (41)
    # ── head / face (blue) ───────────────────────────────────────────────────
    ( 1,  2, (255, 153,  51)),  # left_eye    → right_eye
    ( 0,  1, (255, 153,  51)),  # nose        → left_eye
    ( 0,  2, (255, 153,  51)),  # nose        → right_eye
    ( 1,  3, (255, 153,  51)),  # left_eye    → left_ear
    ( 2,  4, (255, 153,  51)),  # right_eye   → right_ear
    ( 3,  5, (255, 153,  51)),  # left_ear    → left_shoulder
    ( 4,  6, (255, 153,  51)),  # right_ear   → right_shoulder
    # ── feet (left=green, right=orange) ──────────────────────────────────────
    (13, 15, (  0, 255,   0)),  # left_ankle  → left_big_toe
    (13, 16, (  0, 255,   0)),  # left_ankle  → left_small_toe
    (13, 17, (  0, 255,   0)),  # left_ankle  → left_heel
    (14, 18, (  0, 128, 255)),  # right_ankle → right_big_toe
    (14, 19, (  0, 128, 255)),  # right_ankle → right_small_toe
    (14, 20, (  0, 128, 255)),  # right_ankle → right_heel
    # ── left hand ─────────────────────────────────────────────────────────────
    # thumb (orange)
    (62, 45, (  0, 128, 255)),  # left_wrist → left_thumb_third_joint (45)
    (45, 44, (  0, 128, 255)),  # → left_thumb2 (44)
    (44, 43, (  0, 128, 255)),  # → left_thumb3 (43)
    (43, 42, (  0, 128, 255)),  # → left_thumb4 (42)
    # index/forefinger (pink)
    (62, 49, (255, 153, 255)),  # left_wrist → left_forefinger_third_joint (49)
    (49, 48, (255, 153, 255)),  # → left_forefinger2 (48)
    (48, 47, (255, 153, 255)),  # → left_forefinger3 (47)
    (47, 46, (255, 153, 255)),  # → left_forefinger4 (46)
    # middle (light blue)
    (62, 53, (255, 178, 102)),  # left_wrist → left_middle_finger_third_joint (53)
    (53, 52, (255, 178, 102)),  # → left_middle_finger2 (52)
    (52, 51, (255, 178, 102)),  # → left_middle_finger3 (51)
    (51, 50, (255, 178, 102)),  # → left_middle_finger4 (50)
    # ring (red)
    (62, 57, ( 51,  51, 255)),  # left_wrist → left_ring_finger_third_joint (57)
    (57, 56, ( 51,  51, 255)),  # → left_ring_finger2 (56)
    (56, 55, ( 51,  51, 255)),  # → left_ring_finger3 (55)
    (55, 54, ( 51,  51, 255)),  # → left_ring_finger4 (54)
    # pinky (green)
    (62, 61, (  0, 255,   0)),  # left_wrist → left_pinky_finger_third_joint (61)
    (61, 60, (  0, 255,   0)),  # → left_pinky_finger2 (60)
    (60, 59, (  0, 255,   0)),  # → left_pinky_finger3 (59)
    (59, 58, (  0, 255,   0)),  # → left_pinky_finger4 (58)
    # ── right hand ────────────────────────────────────────────────────────────
    # thumb (orange)
    (41, 24, (  0, 128, 255)),  # right_wrist → right_thumb_third_joint (24)
    (24, 23, (  0, 128, 255)),  # → right_thumb2 (23)
    (23, 22, (  0, 128, 255)),  # → right_thumb3 (22)
    (22, 21, (  0, 128, 255)),  # → right_thumb4 (21)
    # index/forefinger (pink)
    (41, 28, (255, 153, 255)),  # right_wrist → right_forefinger_third_joint (28)
    (28, 27, (255, 153, 255)),  # → right_forefinger2 (27)
    (27, 26, (255, 153, 255)),  # → right_forefinger3 (26)
    (26, 25, (255, 153, 255)),  # → right_forefinger4 (25)
    # middle (light blue)
    (41, 32, (255, 178, 102)),  # right_wrist → right_middle_finger_third_joint (32)
    (32, 31, (255, 178, 102)),  # → right_middle_finger2 (31)
    (31, 30, (255, 178, 102)),  # → right_middle_finger3 (30)
    (30, 29, (255, 178, 102)),  # → right_middle_finger4 (29)
    # ring (red)
    (41, 36, ( 51,  51, 255)),  # right_wrist → right_ring_finger_third_joint (36)
    (36, 35, ( 51,  51, 255)),  # → right_ring_finger2 (35)
    (35, 34, ( 51,  51, 255)),  # → right_ring_finger3 (34)
    (34, 33, ( 51,  51, 255)),  # → right_ring_finger4 (33)
    # pinky (green)
    (41, 40, (  0, 255,   0)),  # right_wrist → right_pinky_finger_third_joint (40)
    (40, 39, (  0, 255,   0)),  # → right_pinky_finger2 (39)
    (39, 38, (  0, 255,   0)),  # → right_pinky_finger3 (38)
    (38, 37, (  0, 255,   0)),  # → right_pinky_finger4 (37)
]


def _render_3d_opencv(outputs: list, canvas_hw: tuple) -> np.ndarray:
    """
    Project 3-D keypoints onto a 2-D OpenCV canvas (front view: X right, -Y up).
    Returns a BGR uint8 image of shape (canvas_hw[0], canvas_hw[1], 3).
    """
    h, w = canvas_hw
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    lean_angles: list[float] = []

    if not outputs:
        cv2.putText(canvas, "No person detected", (20, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        return canvas, lean_angles

    # Dim multipliers per person so multiple people are visually distinct
    brightness = [1.0, 0.6, 0.8, 0.5]

    for person_idx, person in enumerate(outputs):
        kp3d = person.get("pred_keypoints_3d")
        if kp3d is None:
            continue
        if hasattr(kp3d, "cpu"):
            kp3d = kp3d.cpu().numpy()
        kp3d = np.array(kp3d, dtype=np.float32)
        if kp3d.ndim != 2 or kp3d.shape[1] < 3:
            continue

        n_joints = kp3d.shape[0]
        xs =  kp3d[:, 2]  # side view: Z (depth) → horizontal; front of body on left
        ys =  kp3d[:, 1]  # camera +Y is already down = pixel down

        # Fixed metric scale: anchor on pelvis (midpoint of left_hip=9, right_hip=10).
        HALF_EXTENT = 1.0  # metres shown from centre to canvas edge
        pelvis_x = (kp3d[9, 2] + kp3d[10, 2]) / 2
        pelvis_y = (kp3d[9, 1] + kp3d[10, 1]) / 2
        scale = min(w, h) * 0.45 / HALF_EXTENT  # pixels per metre
        px = (xs - pelvis_x) * scale + w / 2
        py = (ys - pelvis_y) * scale + h / 2
        pts = np.stack([px, py], axis=1).astype(int)

        dim = brightness[person_idx % len(brightness)]

        # Draw bones with per-bone colors from mhr70.py skeleton_info
        for a, b, bone_col in _BONES:
            if a < n_joints and b < n_joints:
                col = tuple(int(c * dim) for c in bone_col)
                cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), col, 2, cv2.LINE_AA)

        # Draw joints as white dots so they stand out against any bone color
        for j in range(n_joints):
            cv2.circle(canvas, tuple(pts[j]), 3, (220, 220, 220), -1, cv2.LINE_AA)

        # Label
        cv2.putText(canvas, f"P{person_idx}", (10, 25 + person_idx * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

        # ── Forward lean angle ───────────────────────────────────────────────
        # Spine vector: pelvis midpoint → neck (joint 69).
        # Angle is measured in the sagittal (Y-Z) plane.
        # +Y is down, +Z is away from camera → leaning toward camera = -Z,
        # so lean = arctan2(-spine_z, -spine_y) where -spine_y is the upward component.
        if n_joints > 69:
            pelvis_3d  = (kp3d[9]  + kp3d[10]) / 2
            neck_3d    = kp3d[69]
            spine      = neck_3d - pelvis_3d          # points upward & possibly forward
            lean_deg   = np.degrees(np.arctan2(-spine[2], -spine[1]))
            # positive = leaning toward camera (forward), negative = leaning back

            # Draw reference vertical line at pelvis canvas position
            pelvis_pt  = ((pts[9]  + pts[10]) / 2).astype(int)
            ref_top    = (pelvis_pt[0], pelvis_pt[1] - 80)
            cv2.line(canvas, tuple(pelvis_pt), ref_top, (80, 80, 80), 1, cv2.LINE_AA)

            # Draw actual spine direction line
            neck_pt    = tuple(pts[69])
            cv2.line(canvas, tuple(pelvis_pt), neck_pt, (0, 255, 255), 2, cv2.LINE_AA)

            # Angle text
            lean_angles.append(lean_deg)
            label = f"Lean: {lean_deg:+.1f} deg"
            col   = (0, 200, 255) if abs(lean_deg) < 15 else (0, 80, 255)
            cv2.putText(canvas, label,
                        (10, h - 30 - person_idx * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

    cv2.putText(canvas, "3D skeleton (side view, front=left)", (5, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    return canvas, lean_angles


def run_webcam(estimator, args: argparse.Namespace):
    """
    Real-time webcam demo.
    - Capture thread reads frames from the camera and drops them into an input queue.
    - Inference thread (this function) pops frames, runs the model, and pushes results.
    - Display loop shows the latest annotated frame side-by-side with the 3D view.
    Press Q to quit.
    """
    logger.info(f"Opening webcam device {args.webcam_id} …")
    cap = cv2.VideoCapture(args.webcam_id)
    if not cap.isOpened():
        logger.error(f"Cannot open webcam device {args.webcam_id}")
        return

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    logger.info(f"Camera: {cam_w}×{cam_h}")

    # Queues: raw frames in → inference → annotated frames out
    in_q: queue.Queue  = queue.Queue(maxsize=1)   # drop stale frames
    out_q: queue.Queue = queue.Queue(maxsize=1)

    stop_event = threading.Event()
    visualizer = setup_visualizer()

    # ── Capture thread ──────────────────────────────────────────────────────
    def _capture():
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                stop_event.set()
                break
            try:
                in_q.put_nowait(frame)
            except queue.Full:
                pass  # drop frame — inference can't keep up

    capture_thread = threading.Thread(target=_capture, daemon=True)
    capture_thread.start()

    # ── Inference thread ────────────────────────────────────────────────────
    def _infer():
        frame_n = 0
        # Rolling timing accumulators (seconds)
        t_write = t_infer = t_vis = 0.0
        REPORT_EVERY = 5

        while not stop_event.is_set():
            try:
                frame = in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            # ① convert BGR→RGB (estimator expects RGB for numpy input; no disk write)
            t0 = time.perf_counter()
            frame_rgb = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t_write += time.perf_counter() - t0

            # ② model inference
            # Patch out the per-frame empty_cache() — it stalls the GPU pipeline
            # and is unnecessary in a continuous loop.
            t0 = time.perf_counter()
            try:
                _real_empty_cache = torch.cuda.empty_cache
                torch.cuda.empty_cache = lambda: None
                with _quiet(), _autocast_ctx(args):
                    outputs = estimator.process_one_image(
                        frame_rgb,
                        inference_type="body" if args.body_only else "full",
                    )
            except Exception as exc:
                logger.warning(f"Inference error: {exc}")
                outputs = None
            finally:
                torch.cuda.empty_cache = _real_empty_cache
            t_infer += time.perf_counter() - t0

            # ③ 2D visualisation
            t0 = time.perf_counter()
            if outputs:
                vis_list = visualize_2d_results(frame, outputs, visualizer)
                annotated = vis_list[0] if vis_list else frame
            else:
                annotated = frame
                outputs = []
            t_vis += time.perf_counter() - t0

            frame_n += 1

            # Print timing summary every REPORT_EVERY frames
            if frame_n % REPORT_EVERY == 0:
                n = REPORT_EVERY
                total = t_write + t_infer + t_vis
                logger.info(
                    f"[timing/{n} frames]  "
                    f"disk_write={t_write/n*1000:.1f}ms  "
                    f"inference={t_infer/n*1000:.1f}ms  "
                    f"visualize_2d={t_vis/n*1000:.1f}ms  "
                    f"total={total/n*1000:.1f}ms  "
                    f"({n/(total) if total>0 else 0:.2f} fps)"
                )
                t_write = t_infer = t_vis = 0.0

            result = (annotated, outputs)
            try:
                out_q.put_nowait(result)
            except queue.Full:
                try:
                    out_q.get_nowait()  # discard old result
                except queue.Empty:
                    pass
                out_q.put_nowait(result)

    infer_thread = threading.Thread(target=_infer, daemon=True)
    infer_thread.start()

    # ── Display loop ────────────────────────────────────────────────────────
    logger.info("Webcam running — press Q to quit.")

    latest_annotated = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
    latest_outputs: list = []
    skel_h, skel_w = cam_h, cam_h  # square 3-D canvas same height as camera

    LEAN_THRESHOLD  = 8.0   # degrees — applied to 60-second rolling average
    BEEP_INTERVAL   = 3.0   # seconds between beeps
    AVERAGE_WINDOW  = 60.0  # seconds of history to average over
    last_beep_time  = 0.0
    # deque of (timestamp, angle) for the first detected person
    lean_history: collections.deque = collections.deque()

    def _beep():
        import subprocess
        subprocess.Popen(["paplay", "--volume=65536", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cv2.namedWindow("SAM-3D-Body | Webcam", cv2.WINDOW_NORMAL)

    while not stop_event.is_set():
        # Check for a new inference result (non-blocking)
        try:
            latest_annotated, latest_outputs = out_q.get_nowait()
        except queue.Empty:
            pass

        # Resize annotated frame to cam_h if needed
        ann = latest_annotated
        if ann.shape[0] != cam_h:
            ann = cv2.resize(ann, (int(ann.shape[1] * cam_h / ann.shape[0]), cam_h))

        if args.no_3d:
            display = ann
            lean_angles = []
        else:
            skel_canvas, lean_angles = _render_3d_opencv(latest_outputs, (skel_h, skel_w))
            display = np.hstack([ann, skel_canvas])

        # Rolling 60-second average lean check
        now = time.perf_counter()
        if lean_angles:
            lean_history.append((now, lean_angles[0]))
        # Drop samples older than the window
        while lean_history and now - lean_history[0][0] > AVERAGE_WINDOW:
            lean_history.popleft()
        if lean_history:
            avg_lean = sum(a for _, a in lean_history) / len(lean_history)
            secs = min(now - lean_history[0][0], AVERAGE_WINDOW)
            over = abs(avg_lean) > LEAN_THRESHOLD
            avg_col = (0, 80, 255) if over else (0, 220, 120)
            cv2.putText(display,
                        f"Avg lean ({secs:.0f}s): {avg_lean:+.1f} deg",
                        (10, display.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, avg_col, 2, cv2.LINE_AA)
            if over and now - last_beep_time >= BEEP_INTERVAL:
                threading.Thread(target=_beep, daemon=True).start()
                last_beep_time = now
            
            if lean_angles[0] < LEAN_THRESHOLD:
                lean_history.clear()

        cv2.imshow("SAM-3D-Body | Webcam", display)

        if cv2.waitKey(30) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Webcam closed.")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.webcam:
        estimator = load_estimator(args)
        run_webcam(estimator, args)
        return

    if not args.input:
        logger.error("Provide --input <file> or use --webcam for live camera.")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)

    out_dir = REPO_ROOT / "output"
    output_path = Path(args.output) if args.output else out_dir / input_path.name

    estimator = load_estimator(args)

    suffix = input_path.suffix.lower()
    if suffix in IMAGE_EXTS:
        run_image(estimator, input_path, output_path, args)
    elif suffix in VIDEO_EXTS:
        run_video(estimator, input_path, output_path, args)
    else:
        logger.error(
            f"Unsupported extension '{suffix}'.\n"
            f"  Images: {IMAGE_EXTS}\n"
            f"  Videos: {VIDEO_EXTS}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
