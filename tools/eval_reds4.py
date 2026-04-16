# Copyright (c) OpenMMLab. All rights reserved.
"""Offline REDS4 evaluation: PSNR / SSIM / LPIPS / NIQE / tOF / tLP / DISTS / MUSIQ.

Run after `tools/test_reds4.sh` has written SR outputs to disk. The script
walks `<sr_dir>/<clip>/*.png` and matches them against `<gt_dir>/<clip>/*.png`,
reporting per-clip and overall averages.

Metric references:
  - PSNR / SSIM / NIQE : mmedit.core.evaluation.metrics (reused from repo)
  - LPIPS             : Zhang et al., CVPR 2018 (`pip install lpips`)
  - tOF / tLP         : Chu et al., TecoGAN, ACM TOG 2020
      tOF = mean L2 of (flow(GT_t, GT_{t+1}) - flow(SR_t, SR_{t+1}))
      tLP = |LPIPS(GT_t, GT_{t+1}) - LPIPS(SR_t, SR_{t+1})| (x100)
  - DISTS             : Ding et al., TPAMI 2020 (via `pyiqa`)
  - MUSIQ             : Ke et al., ICCV 2021  (via `pyiqa`, KonIQ-10k)

Install extras: pip install lpips pyiqa

Example:
  python tools/eval_reds4.py \\
    --sr-dir  /mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/test_reds \\
    --gt-dir  /mnt/HDD_raid1/yjcho/data/REDS/test/gt \\
    --metrics PSNR SSIM LPIPS NIQE tOF tLP DISTS MUSIQ \\
    --device  cuda:0
"""

import argparse
import os
import os.path as osp
import sys
import time
from collections import OrderedDict

import cv2
import numpy as np

# Ensure repo root on path so we can reuse mmedit metric implementations
_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mmedit.core.evaluation.metrics import psnr as _psnr  # noqa: E402
from mmedit.core.evaluation.metrics import ssim as _ssim  # noqa: E402
from mmedit.core.evaluation.metrics import niqe as _niqe  # noqa: E402


REDS4_CLIPS = ('000', '011', '015', '020')
ALL_METRICS = ('PSNR', 'SSIM', 'LPIPS', 'NIQE', 'tOF', 'tLP', 'DISTS', 'MUSIQ')


def parse_args():
    p = argparse.ArgumentParser(description='REDS4 offline evaluation')
    p.add_argument('--sr-dir', required=True,
                   help='root of SR outputs; expects <sr_dir>/<clip>/*.png')
    p.add_argument('--gt-dir', required=True,
                   help='root of GT frames;   expects <gt_dir>/<clip>/*.png')
    p.add_argument('--metrics', nargs='+', default=list(ALL_METRICS),
                   choices=list(ALL_METRICS),
                   help='subset of metrics to compute')
    p.add_argument('--clips', nargs='+', default=list(REDS4_CLIPS),
                   help='clip subdirectories to evaluate')
    p.add_argument('--crop-border', type=int, default=0,
                   help='pixels to crop from each border for PSNR/SSIM')
    p.add_argument('--convert-to', default=None, choices=[None, 'y'],
                   help='convert to Y-channel before PSNR/SSIM (default RGB)')
    p.add_argument('--lpips-net', default='alex', choices=['alex', 'vgg'],
                   help='LPIPS backbone (alex matches TecoGAN convention)')
    p.add_argument('--musiq-variant', default='musiq',
                   help='pyiqa MUSIQ variant: musiq (koniq), musiq-ava, '
                        'musiq-spaq, musiq-paq2piq')
    p.add_argument('--device', default='cuda:0',
                   help='device for deep metrics (e.g. cuda:0, cuda:2, cpu)')
    return p.parse_args()


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------

def list_frames(clip_dir):
    frames = sorted(f for f in os.listdir(clip_dir) if f.lower().endswith(
        ('.png', '.jpg', '.jpeg')))
    return [osp.join(clip_dir, f) for f in frames]


def load_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'cv2.imread failed: {path}')
    return img


# --------------------------------------------------------------------------
# Metric wrappers
# --------------------------------------------------------------------------

def metric_psnr(sr, gt, crop_border, convert_to):
    return _psnr(sr, gt, crop_border=crop_border,
                 input_order='HWC', convert_to=convert_to)


def metric_ssim(sr, gt, crop_border, convert_to):
    return _ssim(sr, gt, crop_border=crop_border,
                 input_order='HWC', convert_to=convert_to)


def metric_niqe(sr, crop_border):
    # NIQE is no-reference; operates on SR only, Y-channel as in mmedit.
    return _niqe(sr, crop_border=crop_border,
                 input_order='HWC', convert_to='y')


def bgr_to_lpips_tensor(img_bgr, device):
    """uint8 BGR HxWxC -> float RGB 1x3xHxW in [-1, 1] on device."""
    import torch
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float()
    t = t / 127.5 - 1.0
    return t.unsqueeze(0)


def bgr_to_01_tensor(img_bgr, device):
    """uint8 BGR HxWxC -> float RGB 1x3xHxW in [0, 1] on device (pyiqa range)."""
    import torch
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def farneback_flow(img_bgr_a, img_bgr_b):
    ga = cv2.cvtColor(img_bgr_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(img_bgr_b, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        ga, gb, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    need_lpips = 'LPIPS' in args.metrics or 'tLP' in args.metrics
    need_flow = 'tOF' in args.metrics

    lpips_model = None
    if need_lpips:
        try:
            import lpips  # noqa: F401
            import torch
        except ImportError:
            print('[!] LPIPS/tLP requested but `lpips` not installed. '
                  'Install with: pip install lpips')
            sys.exit(1)
        lpips_model = lpips.LPIPS(net=args.lpips_net).to(args.device)
        lpips_model.eval()

    need_dists = 'DISTS' in args.metrics
    need_musiq = 'MUSIQ' in args.metrics
    dists_model = musiq_model = None
    if need_dists or need_musiq:
        try:
            import pyiqa  # noqa: F401
            import torch
        except ImportError:
            print('[!] DISTS/MUSIQ requested but `pyiqa` not installed. '
                  'Install with: pip install pyiqa')
            sys.exit(1)
        if need_dists:
            dists_model = pyiqa.create_metric('dists', device=args.device)
            dists_model.eval()
        if need_musiq:
            musiq_model = pyiqa.create_metric(args.musiq_variant,
                                              device=args.device)
            musiq_model.eval()

    # NIQE impl reads a relative .npz; switch cwd to repo root just in case.
    os.chdir(_REPO_ROOT)

    per_clip = OrderedDict()

    for clip in args.clips:
        sr_clip = osp.join(args.sr_dir, clip)
        gt_clip = osp.join(args.gt_dir, clip)
        if not osp.isdir(sr_clip):
            print(f'[skip] SR clip missing: {sr_clip}')
            continue
        if not osp.isdir(gt_clip):
            print(f'[skip] GT clip missing: {gt_clip}')
            continue

        sr_files = list_frames(sr_clip)
        gt_files = list_frames(gt_clip)
        if len(sr_files) != len(gt_files):
            print(f'[warn] {clip}: #SR={len(sr_files)} != #GT={len(gt_files)}; '
                  f'truncating to min.')
            n = min(len(sr_files), len(gt_files))
            sr_files, gt_files = sr_files[:n], gt_files[:n]

        acc = {m: [] for m in args.metrics}

        # Keep previous frame in memory to avoid double reads for tOF / tLP.
        prev_sr = prev_gt = None
        prev_sr_t = prev_gt_t = None  # lpips tensors

        t0 = time.time()
        for idx, (sr_path, gt_path) in enumerate(zip(sr_files, gt_files)):
            sr = load_bgr(sr_path)
            gt = load_bgr(gt_path)
            if sr.shape != gt.shape:
                raise RuntimeError(
                    f'Shape mismatch at {sr_path} vs {gt_path}: '
                    f'{sr.shape} != {gt.shape}')

            if 'PSNR' in acc:
                acc['PSNR'].append(metric_psnr(
                    sr, gt, args.crop_border, args.convert_to))
            if 'SSIM' in acc:
                acc['SSIM'].append(metric_ssim(
                    sr, gt, args.crop_border, args.convert_to))
            if 'NIQE' in acc:
                acc['NIQE'].append(metric_niqe(sr, args.crop_border))

            sr_t = gt_t = None
            if need_lpips:
                import torch
                with torch.no_grad():
                    sr_t = bgr_to_lpips_tensor(sr, args.device)
                    gt_t = bgr_to_lpips_tensor(gt, args.device)
                    if 'LPIPS' in acc:
                        d = lpips_model(sr_t, gt_t).item()
                        acc['LPIPS'].append(d)

            if need_dists or need_musiq:
                import torch
                with torch.no_grad():
                    sr_01 = bgr_to_01_tensor(sr, args.device)
                    if 'DISTS' in acc:
                        gt_01 = bgr_to_01_tensor(gt, args.device)
                        acc['DISTS'].append(float(
                            dists_model(sr_01, gt_01).item()))
                    if 'MUSIQ' in acc:
                        acc['MUSIQ'].append(float(
                            musiq_model(sr_01).item()))

            if idx > 0:
                if 'tOF' in acc:
                    flow_sr = farneback_flow(prev_sr, sr)
                    flow_gt = farneback_flow(prev_gt, gt)
                    acc['tOF'].append(float(np.mean(
                        np.sqrt(np.sum((flow_sr - flow_gt) ** 2, axis=-1)))))
                if 'tLP' in acc and need_lpips:
                    import torch
                    with torch.no_grad():
                        lp_sr = lpips_model(prev_sr_t, sr_t).item()
                        lp_gt = lpips_model(prev_gt_t, gt_t).item()
                    acc['tLP'].append(abs(lp_sr - lp_gt) * 100.0)

            prev_sr, prev_gt = sr, gt
            prev_sr_t, prev_gt_t = sr_t, gt_t

        clip_res = {m: (float(np.mean(v)) if v else float('nan'))
                    for m, v in acc.items()}
        per_clip[clip] = clip_res
        dt = time.time() - t0
        summary = '  '.join(f'{m}={clip_res[m]:.4f}' for m in args.metrics)
        print(f'[{clip}] frames={len(sr_files)}  {summary}  ({dt:.1f}s)')

    if not per_clip:
        print('No clips evaluated.')
        return

    print('\n=== Average over {} clip(s) ==='.format(len(per_clip)))
    overall = {m: float(np.mean([per_clip[c][m] for c in per_clip]))
               for m in args.metrics}
    for m in args.metrics:
        print(f'{m:>6s}: {overall[m]:.4f}')


if __name__ == '__main__':
    main()
