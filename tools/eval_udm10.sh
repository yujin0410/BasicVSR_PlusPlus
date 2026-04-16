#!/usr/bin/env bash
# Offline UDM10 evaluation using tools/eval_reds4.py with UDM10 clips.
# Assumes `sh tools/test_udm10.sh` has populated SR_DIR/<clip>/*.png.
#
# Override via env vars, e.g.:
#   METRICS="PSNR SSIM" sh tools/eval_udm10.sh

set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

SR_DIR="${SR_DIR:-/mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/test_udm10}"
GT_DIR="${GT_DIR:-/mnt/HDD_raid1/yjcho/data/UDM10/GT}"
METRICS="${METRICS:-PSNR SSIM LPIPS NIQE tOF tLP DISTS MUSIQ CLIPIQA}"
LPIPS_NET="${LPIPS_NET:-alex}"
DEVICE="${DEVICE:-cuda:0}"

CLIPS="${CLIPS:-000 001 002 003 004 005 006 007 008 009}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

echo "[eval_udm10] CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo "[eval_udm10] SR_DIR    : ${SR_DIR}"
echo "[eval_udm10] GT_DIR    : ${GT_DIR}"
echo "[eval_udm10] METRICS   : ${METRICS}"
echo "[eval_udm10] CLIPS     : ${CLIPS}"
echo "[eval_udm10] LPIPS_NET : ${LPIPS_NET}"
echo "[eval_udm10] DEVICE    : ${DEVICE}"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
python "${REPO_ROOT}/tools/eval_reds4.py" \
    --sr-dir "${SR_DIR}" \
    --gt-dir "${GT_DIR}" \
    --clips ${CLIPS} \
    --metrics ${METRICS} \
    --lpips-net "${LPIPS_NET}" \
    --device "${DEVICE}"
