"""
Stabilizer 수치 벤치마크 (헤드리스)

측정 지표:
  1. frame_diff_bg   : 배경 ROI의 프레임 간 픽셀 변화량 평균
                       → 낮을수록 배경이 안정 = 안정화 효과 있음
  2. frame_diff_all  : 전체 화면 프레임 간 픽셀 변화량 평균
  3. jitter_rms      : RANSAC이 측정한 raw 이동량의 RMS
                       → 카메라가 얼마나 떨렸는가
  4. inlier_ratio    : RANSAC 인라이어 비율 (낮으면 움직이는 물체가 많음)
  5. smooth_dx/dy    : EMA 스무딩 후 실제 적용된 보정량

배경 ROI: 화면 네 귀퉁이 (밥솥/딸랑이가 없는 영역으로 가정)

실행:
  cd c:\\Solo\\Pressure_Chicken
  uv run python tests/bench_stabilizer.py
  uv run python tests/bench_stabilizer.py --video raw/Sample02.mp4 --frames 300
"""

from __future__ import annotations

import argparse
import json
import sys
import os

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.stabilizer import Stabilizer

parser = argparse.ArgumentParser()
parser.add_argument("--video",  default="raw/Sample01.mp4")
parser.add_argument("--frames", type=int, default=300)
args = parser.parse_args()

with open("config/store_config.json", encoding="utf-8") as f:
    cfg = json.load(f)
stab_cfg = cfg.get("stabilizer", {})

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"[ERROR] Cannot open: {args.video}")
    sys.exit(1)

fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"\n[Video] {args.video}  {vid_w}x{vid_h}  {fps:.1f}fps")
print(f"[Config] grid={stab_cfg.get('grid_rows')}x{stab_cfg.get('grid_cols')}  "
      f"alpha={stab_cfg.get('smooth_alpha')}  ransac_thr={stab_cfg.get('ransac_threshold')}\n")

stab = Stabilizer(stab_cfg)

# 배경 ROI: 귀퉁이 4곳 (각 10% 크기)
bx, by = int(vid_w * 0.1), int(vid_h * 0.1)
bg_rois = [
    (0,           0,           bx, by),   # 좌상
    (vid_w - bx,  0,           vid_w, by), # 우상
    (0,           vid_h - by,  bx, vid_h), # 좌하
    (vid_w - bx,  vid_h - by,  vid_w, vid_h), # 우하
]

prev_orig_gray: np.ndarray | None = None
prev_stab_gray: np.ndarray | None = None

# 누적 통계
stats = {
    "orig_bg_diff":   [],
    "stab_bg_diff":   [],
    "orig_all_diff":  [],
    "stab_all_diff":  [],
    "raw_dx":         [],
    "raw_dy":         [],
    "smooth_dx":      [],
    "smooth_dy":      [],
    "inlier_ratio":   [],
}

frame_idx = 0
print(f"{'frame':>6}  {'raw_dx':>7} {'raw_dy':>7}  "
      f"{'inlier%':>8}  {'orig_bg':>8} {'stab_bg':>8}  {'diff%':>7}")
print("-" * 72)

while frame_idx < args.frames:
    ret, frame = cap.read()
    if not ret:
        break

    stabilized = stab.stabilize(frame)

    orig_gray = cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY).astype(np.float32)
    stab_gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if prev_orig_gray is not None and prev_stab_gray is not None:
        # 전체 화면 프레임 차이
        orig_all_diff = float(np.mean(np.abs(orig_gray - prev_orig_gray)))
        stab_all_diff = float(np.mean(np.abs(stab_gray - prev_stab_gray)))

        # 배경 ROI 프레임 차이
        orig_bg_diffs, stab_bg_diffs = [], []
        for x1, y1, x2, y2 in bg_rois:
            orig_bg_diffs.append(float(np.mean(np.abs(
                orig_gray[y1:y2, x1:x2] - prev_orig_gray[y1:y2, x1:x2]))))
            stab_bg_diffs.append(float(np.mean(np.abs(
                stab_gray[y1:y2, x1:x2] - prev_stab_gray[y1:y2, x1:x2]))))

        orig_bg = float(np.mean(orig_bg_diffs))
        stab_bg = float(np.mean(stab_bg_diffs))
        improvement = (orig_bg - stab_bg) / orig_bg * 100 if orig_bg > 0 else 0.0

        inlier_r = stab.last_n_inliers / stab.last_n_features if stab.last_n_features > 0 else 0.0

        stats["orig_bg_diff"].append(orig_bg)
        stats["stab_bg_diff"].append(stab_bg)
        stats["orig_all_diff"].append(orig_all_diff)
        stats["stab_all_diff"].append(stab_all_diff)
        stats["raw_dx"].append(stab.last_raw_dx)
        stats["raw_dy"].append(stab.last_raw_dy)
        stats["smooth_dx"].append(stab.last_smooth_dx)
        stats["smooth_dy"].append(stab.last_smooth_dy)
        stats["inlier_ratio"].append(inlier_r)

        if frame_idx % 30 == 0:
            print(f"{frame_idx:>6}  {stab.last_raw_dx:>+7.2f} {stab.last_raw_dy:>+7.2f}  "
                  f"{inlier_r*100:>7.1f}%  "
                  f"{orig_bg:>8.3f} {stab_bg:>8.3f}  "
                  f"{improvement:>+6.1f}%")

    prev_orig_gray = orig_gray
    prev_stab_gray = stab_gray
    frame_idx += 1

cap.release()

# ── 최종 요약 ────────────────────────────────────────────────────────────────
def rms(lst): return float(np.sqrt(np.mean(np.array(lst)**2))) if lst else 0.0
def mean(lst): return float(np.mean(lst)) if lst else 0.0

print("\n" + "=" * 72)
print(f"[결과 요약]  {frame_idx}프레임 분석\n")

orig_bg_mean = mean(stats["orig_bg_diff"])
stab_bg_mean = mean(stats["stab_bg_diff"])
bg_improve   = (orig_bg_mean - stab_bg_mean) / orig_bg_mean * 100 if orig_bg_mean > 0 else 0.0

orig_all_mean = mean(stats["orig_all_diff"])
stab_all_mean = mean(stats["stab_all_diff"])
all_improve   = (orig_all_mean - stab_all_mean) / orig_all_mean * 100 if orig_all_mean > 0 else 0.0

jitter_rms = rms(stats["raw_dx"]) + rms(stats["raw_dy"])

print(f"  카메라 떨림 (raw 이동량 RMS)  : dx={rms(stats['raw_dx']):.3f}px  dy={rms(stats['raw_dy']):.3f}px")
print(f"  EMA 보정량 (smooth)           : dx={rms(stats['smooth_dx']):.3f}px  dy={rms(stats['smooth_dy']):.3f}px")
print(f"  RANSAC 인라이어 비율 평균     : {mean(stats['inlier_ratio'])*100:.1f}%")
print()
print(f"  배경 픽셀 변화량  원본={orig_bg_mean:.4f}  안정화={stab_bg_mean:.4f}  개선={bg_improve:+.1f}%")
print(f"  전체 픽셀 변화량  원본={orig_all_mean:.4f}  안정화={stab_all_mean:.4f}  개선={all_improve:+.1f}%")
print()

if jitter_rms < 0.5:
    print("  [verdict] Camera shake near zero -- stabilization effect minimal on this video")
elif bg_improve > 5:
    print(f"  [verdict] Background stabilized -- {bg_improve:.1f}% improvement")
else:
    print("  [verdict] Minimal improvement -- tune params or video has little shake")

print("=" * 72)
