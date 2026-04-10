#!/usr/bin/env bash
# SAM-3D-Body setup script
# Run once to clone the model repo, install detectron2, and authenticate with HuggingFace.
#
# Prerequisites:
#   - CUDA toolkit (sudo pacman -S cuda)
#   - uv (https://docs.astral.sh/uv/getting-started/installation/)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="$SCRIPT_DIR/vendor"
CUDA_DIR="${CUDA_HOME:-/opt/cuda}"

echo "=== SAM-3D-Body Demo Setup ==="
echo ""

# ── 1. Sync uv environment (installs PyTorch cu126 + all deps) ────────────────
echo "[1/5] Syncing uv environment (PyTorch 2.10 + cu126)..."
uv sync

# ── 2. Clone sam-3d-body ──────────────────────────────────────────────────────
mkdir -p "$VENDOR_DIR"
if [ ! -d "$VENDOR_DIR/sam-3d-body" ]; then
    echo ""
    echo "[2/5] Cloning facebookresearch/sam-3d-body..."
    git clone https://github.com/facebookresearch/sam-3d-body.git "$VENDOR_DIR/sam-3d-body"
else
    echo "[2/5] sam-3d-body already present — skipping clone."
fi

# ── 3. Install ninja (faster CUDA build backend) ─────────────────────────────
echo ""
echo "[3/5] Installing ninja build backend..."
uv pip install ninja

# ── 4. Clone and build detectron2 (pinned commit) ─────────────────────────────
echo ""
echo "[4/5] Building detectron2 (this takes ~1 minute)..."
if [ ! -d "$VENDOR_DIR/detectron2" ]; then
    git clone https://github.com/facebookresearch/detectron2.git "$VENDOR_DIR/detectron2"
    git -C "$VENDOR_DIR/detectron2" checkout a1ce2f9
else
    echo "      detectron2 already cloned."
fi

CUDA_HOME="$CUDA_DIR" CUDA_PATH="$CUDA_DIR" \
    uv pip install "$VENDOR_DIR/detectron2" --no-build-isolation --no-deps

# ── 5. (Optional) MoGe for FOV estimation ────────────────────────────────────
echo ""
read -r -p "[5/5] Install MoGe (improves FOV estimation, ~1 GB extra)? [y/N] " install_moge
if [[ "$install_moge" =~ ^[Yy]$ ]]; then
    uv pip install "git+https://github.com/microsoft/MoGe.git"
    echo "      MoGe installed."
else
    echo "      Skipping MoGe (use --no-fov flag when running demo.py)."
fi

echo ""
echo "=== Core setup complete! ==="
echo ""
echo "FINAL STEP — HuggingFace authentication:"
echo "  1. Request model access: https://huggingface.co/facebook/sam-3d-body-dinov3"
echo "  2. Then run:  uv run huggingface-cli login"
echo ""
echo "Usage (after login + first model download ~840 MB):"
echo "  uv run demo.py --input path/to/image.jpg --show"
echo "  uv run demo.py --input path/to/video.mp4 --output result.mp4"
echo "  uv run demo.py --input path/to/image.jpg --no-fov   # if MoGe not installed"
