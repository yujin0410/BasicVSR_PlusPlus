#!/usr/bin/env bash
# Offline REDS4 evaluation (PSNR / SSIM / LPIPS / NIQE / tOF / tLP).
# Assumes `sh tools/test_reds4.sh` has already been run and SR frames
# are under SR_DIR/{000,011,015,020}/*.png.
#
# Override paths or metrics via env vars, e.g.:
#   METRICS="PSNR SSIM" sh tools/eval_reds4.sh
#   SR_DIR=/other/path sh tools/eval_reds4.sh

set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

SR_DIR="${SR_DIR:-/mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/test_reds}"
GT_DIR="${GT_DIR:-/mnt/HDD_raid1/yjcho/data/REDS/test/gt}"
METRICS="${METRICS:-PSNR SSIM LPIPS NIQE tOF tLP DISTS MUSIQ CLIPIQA}"
LPIPS_NET="${LPIPS_NET:-alex}"
DEVICE="${DEVICE:-cuda:0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

echo "[eval_reds4] CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo "[eval_reds4] SR_DIR    : ${SR_DIR}"
echo "[eval_reds4] GT_DIR    : ${GT_DIR}"
echo "[eval_reds4] METRICS   : ${METRICS}"
echo "[eval_reds4] LPIPS_NET : ${LPIPS_NET}"
echo "[eval_reds4] DEVICE    : ${DEVICE}"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
python "${REPO_ROOT}/tools/eval_reds4.py" \
    --sr-dir "${SR_DIR}" \
    --gt-dir "${GT_DIR}" \
    --metrics ${METRICS} \
    --lpips-net "${LPIPS_NET}" \
    --device "${DEVICE}"
