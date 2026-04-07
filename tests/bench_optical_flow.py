"""
Phase 2 - Optical Flow 헤드리스 벤치마크

YOLO로 딸랑이 bbox를 먼저 뽑고, 해당 ROI 안에서
Farneback optical flow RMS를 측정해 움직임 판별.

출력:
  - 프레임별 RMS 수치
  - 정지 구간 vs 움직임 구간 RMS 비교
  - 적정 rms_threshold 추천

실행:
  cd c:\\Solo\\Pressure_Chicken
  uv run python tests/bench_optical_flow.py
  uv run python tests/bench_optical_flow.py --frames 500
"""

from __future__ import annotations

import argparse
import json
import sys
import os

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.stabilizer    import Stabilizer
from core.detector      import BurnerDetector, CLASS_POT_WEIGHT
from core.optical_flow  import OpticalFlowDetector

parser = argparse.ArgumentParser()
parser.add_argument("--video",  default="raw/Sample01.mp4")
parser.add_argument("--frames", type=int, default=500)
args = parser.parse_args()

with open("config/store_config.json", encoding="utf-8") as f:
    cfg = json.load(f)

model_cfg = cfg.get("model", {})
stab  = Stabilizer(cfg.get("stabilizer", {}))
det   = BurnerDetector(model_cfg.get("weights", "models/pot_seg.pt"),
                       model_cfg.get("confidence", 0.5))
oflow = OpticalFlowDetector(cfg.get("optical_flow", {}))

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"[ERROR] Cannot open: {args.video}")
    sys.exit(1)

fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"\n[Video] {args.video}  {vid_w}x{vid_h}  {fps:.1f}fps")
print(f"[Config] rms_threshold={cfg['optical_flow']['rms_threshold']}  "
      f"window={cfg['optical_flow']['window_frames']}  "
      f"trigger={cfg['optical_flow']['trigger_frames']}\n")

print(f"{'frame':>6}  {'w_box':>22}  {'rms':>7}  {'skip':>5}  {'motion':>7}  {'of_score':>9}")
print("-" * 72)

rms_log = []
frame_idx = 0

while frame_idx < args.frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Phase 1: 안정화
    stabilized = stab.stabilize(frame)

    # YOLO: 딸랑이 bbox 검출 (전체 프레임)
    dets    = det.detect(stabilized)
    weights = [d for d in dets if d.class_id == CLASS_POT_WEIGHT and d.confidence >= 0.25]

    # 가장 confidence 높은 딸랑이 하나만 사용
    w_box = None
    if weights:
        best = max(weights, key=lambda d: d.confidence)
        w_box = (int(best.x1), int(best.y1), int(best.x2), int(best.y2))

    # Phase 2: Optical Flow
    motion, rms = oflow.update(stabilized, w_box)
    rms_log.append(rms)

    if frame_idx % 15 == 0:
        box_str  = f"({w_box[0]},{w_box[1]},{w_box[2]},{w_box[3]})" if w_box else "  not detected  "
        skip_str = "SKIP" if oflow.last_skipped else ""
        print(f"{frame_idx:>6}  {box_str:>22}  {rms:>7.3f}  {skip_str:>5}  {'True' if motion else 'False':>7}  {oflow.score:>9.2f}")

    frame_idx += 1

cap.release()

# ── 요약 ──────────────────────────────────────────────────────────────────────
rms_arr = np.array(rms_log)
nonzero = rms_arr[rms_arr > 0]

print("\n" + "=" * 65)
print(f"[Summary]  {frame_idx} frames\n")
print(f"  RMS mean : {float(np.mean(nonzero)):.4f}")
print(f"  RMS std  : {float(np.std(nonzero)):.4f}")
print(f"  RMS min  : {float(np.min(nonzero)):.4f}")
print(f"  RMS max  : {float(np.max(nonzero)):.4f}")
print(f"  RMS p25  : {float(np.percentile(nonzero, 25)):.4f}")
print(f"  RMS p75  : {float(np.percentile(nonzero, 75)):.4f}")
print(f"  RMS p95  : {float(np.percentile(nonzero, 95)):.4f}")
print()

suggested = float(np.percentile(nonzero, 60))
print(f"  [Suggested rms_threshold]  ~{suggested:.2f}")
print(f"  (60th percentile -- adjust up if FP, down if FN)")
print("=" * 65)
