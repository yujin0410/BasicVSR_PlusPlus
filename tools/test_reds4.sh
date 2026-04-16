#!/usr/bin/env bash
# Run BasicVSR++ REDS4 test + PSNR evaluation on a single GPU.
#
# Defaults are set for the IVPL-A6000 server layout but can be overridden
# via environment variables, e.g.:
#
#   LQ_DIR=/path/to/lq GT_DIR=/path/to/gt sh tools/test_reds4.sh
#
# Outputs:
#   - PSNR printed to stdout (Eval-PSNR: xx.xx)
#   - Super-resolved frames written under SAVE_DIR/{000,011,015,020}/

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/basicvsr_plusplus_reds4.py}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/chkpts/basicvsr_plusplus_reds4.pth}"
LQ_DIR="${LQ_DIR:-/mnt/HDD_raid1/yjcho/data/REDS/test/bicubic}"
GT_DIR="${GT_DIR:-/mnt/HDD_raid1/yjcho/data/REDS/test/gt}"
SAVE_DIR="${SAVE_DIR:-/mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/test_reds}"

echo "[test_reds4] CONFIG     : ${CONFIG}"
echo "[test_reds4] CHECKPOINT : ${CHECKPOINT}"
echo "[test_reds4] LQ_DIR     : ${LQ_DIR}"
echo "[test_reds4] GT_DIR     : ${GT_DIR}"
echo "[test_reds4] SAVE_DIR   : ${SAVE_DIR}"

mkdir -p "${SAVE_DIR}"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
python "${REPO_ROOT}/tools/test.py" \
    "${CONFIG}" \
    "${CHECKPOINT}" \
    --save-path "${SAVE_DIR}" \
    --cfg-options \
        data.test.type=SRREDSMultipleGTDataset \
        data.test.lq_folder="${LQ_DIR}" \
        data.test.gt_folder="${GT_DIR}" \
        data.test.num_input_frames=100 \
        data.test.scale=4 \
        data.test.val_partition=REDS4 \
        data.test.test_mode=True
