#!/usr/bin/env bash
# Run BasicVSR++ x4 SR inference on all 10 UDM10 clips (000 .. 009).
#
# Uses the demo entry point (restoration_video_demo.py) one clip at a
# time because UDM10 isn't wired into any mmedit dataset class. Each
# clip is a separate 32-frame recurrent forward pass.
#
# Override via env vars, e.g.:
#   DEVICE=0 SAVE_DIR=/other/out sh tools/test_udm10.sh

set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/basicvsr_plusplus_reds4.py}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/chkpts/basicvsr_plusplus_reds4.pth}"
LQ_ROOT="${LQ_ROOT:-/mnt/HDD_raid1/yjcho/data/UDM10/BIx4}"
SAVE_DIR="${SAVE_DIR:-/mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/test_udm10}"
# Note: do not inline "{:04d}.png" inside ${VAR:-default}, bash closes the
# parameter expansion on the first "}" and mangles the template.
if [ -z "${FILENAME_TMPL:-}" ]; then
    FILENAME_TMPL='{:04d}.png'
fi
DEVICE_ID="${DEVICE_ID:-0}"   # seen-id after CUDA_VISIBLE_DEVICES masking

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

CLIPS="${CLIPS:-000 001 002 003 004 005 006 007 008 009}"

echo "[test_udm10] CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES}"
echo "[test_udm10] CONFIG               : ${CONFIG}"
echo "[test_udm10] CHECKPOINT           : ${CHECKPOINT}"
echo "[test_udm10] LQ_ROOT              : ${LQ_ROOT}"
echo "[test_udm10] SAVE_DIR             : ${SAVE_DIR}"
echo "[test_udm10] FILENAME_TMPL        : ${FILENAME_TMPL}"
echo "[test_udm10] CLIPS                : ${CLIPS}"

mkdir -p "${SAVE_DIR}"

for clip in ${CLIPS}; do
    in_dir="${LQ_ROOT}/${clip}"
    out_dir="${SAVE_DIR}/${clip}"
    if [ ! -d "${in_dir}" ]; then
        echo "[test_udm10] skip: missing ${in_dir}"
        continue
    fi
    echo "[test_udm10] >>> ${clip}: ${in_dir} -> ${out_dir}"
    mkdir -p "${out_dir}"
    PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
    python "${REPO_ROOT}/demo/restoration_video_demo.py" \
        "${CONFIG}" \
        "${CHECKPOINT}" \
        "${in_dir}" \
        "${out_dir}" \
        --filename-tmpl "${FILENAME_TMPL}" \
        --device "${DEVICE_ID}"
done

echo "[test_udm10] done."
