#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET_DIR="${ROOT_DIR}/third_party/GroundingDINO"
PYTHON_BIN=${PYTHON_BIN:-python3}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-}
TORCH_VERSION=${TORCH_VERSION:-}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-}

if ! command -v git >/dev/null 2>&1; then
  echo "git is required to clone GroundingDINO." >&2
  exit 1
fi

if [ ! -d "${TARGET_DIR}" ]; then
  echo "Cloning GroundingDINO into ${TARGET_DIR}"
  git clone https://github.com/IDEA-Research/GroundingDINO.git "${TARGET_DIR}"
else
  echo "GroundingDINO already exists at ${TARGET_DIR}"
fi

if command -v uv >/dev/null 2>&1; then
  PIP_CMD=(uv pip)
else
  PIP_CMD=("${PYTHON_BIN}" -m pip)
fi

echo "Installing GroundingDINO dependencies..."
"${PIP_CMD[@]}" install --upgrade pip

if [ -n "${TORCH_VERSION}" ]; then
  TORCH_PKG="torch==${TORCH_VERSION}"
else
  TORCH_PKG="torch"
fi
if [ -n "${TORCHVISION_VERSION}" ]; then
  TORCHVISION_PKG="torchvision==${TORCHVISION_VERSION}"
else
  TORCHVISION_PKG="torchvision"
fi

if [ -n "${TORCH_INDEX_URL}" ]; then
  "${PIP_CMD[@]}" install "${TORCH_PKG}" "${TORCHVISION_PKG}" --index-url "${TORCH_INDEX_URL}"
else
  "${PIP_CMD[@]}" install "${TORCH_PKG}" "${TORCHVISION_PKG}"
fi

"${PIP_CMD[@]}" install -r "${TARGET_DIR}/requirements.txt"
"${PIP_CMD[@]}" install -e "${TARGET_DIR}" --no-build-isolation

echo "\nNext steps:"
cat <<EOF
1) Download a GroundingDINO checkpoint (e.g. groundingdino_swint_ogc.pth)
   and place it under: ${ROOT_DIR}/weights/
2) Provide config + checkpoint to groundingdino, e.g.
   --dino-config ${TARGET_DIR}/groundingdino/config/GroundingDINO_SwinT_OGC.py \
   --dino-checkpoint ${ROOT_DIR}/weights/groundingdino_swint_ogc.pth
3) If you need CUDA wheels, set TORCH_INDEX_URL before running:
   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash scripts/install_grounding_dino.sh
4) If you need a specific torch version (e.g. to avoid build errors), set:
   TORCH_VERSION=2.2.2+cu121 TORCHVISION_VERSION=0.17.2+cu121 \
   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash scripts/install_grounding_dino.sh
EOF
