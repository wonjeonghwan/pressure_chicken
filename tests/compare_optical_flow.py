"""
Phase 2 - Optical Flow 시각적 비교 뷰어

화면 구성:
  ┌─────────────────────┬─────────────────────┐
  │  원본 + YOLO bbox   │  딸랑이 ROI 확대    │
  │  (노란박스=딸랑이)  │  + Flow 색상 오버레이│
  ├─────────────────────┴─────────────────────┤
  │  수치 패널: RMS / skip / motion / score   │
  ├───────────────────────────────────────────┤
  │  그래프: RMS 시계열 + threshold 기준선    │
  └───────────────────────────────────────────┘

Flow 색상 의미:
  색상(Hue) = 이동 방향  /  밝기(Value) = 이동 크기
  어두움 = 거의 안 움직임  /  밝고 화려함 = 많이 움직임

키:
  space : 일시정지 / 재개
  q/ESC : 종료
  → 방향키 : 일시정지 중 한 프레임 앞으로

실행:
  cd c:\\Solo\\Pressure_Chicken
  uv run python tests/compare_optical_flow.py
  uv run python tests/compare_optical_flow.py --video raw/Sample02.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.stabilizer   import Stabilizer
from core.detector     import BurnerDetector, CLASS_POT_WEIGHT
from core.optical_flow import OpticalFlowDetector

# ── 인자 ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--video",     default="raw/Sample01.mp4")
parser.add_argument("--display-w", type=int,   default=1600)
parser.add_argument("--graph-sec", type=float, default=10.0)
args = parser.parse_args()

# ── config ────────────────────────────────────────────────────────────────────
with open("config/store_config.json", encoding="utf-8") as f:
    cfg = json.load(f)
model_cfg = cfg.get("model", {})
flow_cfg  = cfg.get("optical_flow", {})
rms_thr   = flow_cfg.get("rms_threshold", 0.8)

# ── 초기화 ────────────────────────────────────────────────────────────────────
stab  = Stabilizer(cfg.get("stabilizer", {}))
det   = BurnerDetector(model_cfg.get("weights", "models/pot_seg.pt"),
                       model_cfg.get("confidence", 0.5))
oflow = OpticalFlowDetector(flow_cfg)

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"[ERROR] Cannot open: {args.video}")
    sys.exit(1)

fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[Video] {args.video}  {vid_w}x{vid_h}  {total}f  {fps:.1f}fps")
print(f"[Config] rms_threshold={rms_thr}  max_jump={flow_cfg.get('max_box_jump_px',30)}px")
print("  space: pause  ->: step  q/ESC: quit")

# ── 레이아웃 ──────────────────────────────────────────────────────────────────
PANEL_H = 80
GRAPH_H = 100
ROI_W   = 200   # 딸랑이 ROI 확대 표시 크기
ROI_H   = 200

half_w  = (args.display_w - ROI_W) // 2
scale   = half_w / vid_w
disp_h  = int(vid_h * scale)

GRAPH_LEN   = int(fps * args.graph_sec)
rms_hist:    deque[float] = deque([0.0] * GRAPH_LEN, maxlen=GRAPH_LEN)
score_hist:  deque[float] = deque([0.0] * GRAPH_LEN, maxlen=GRAPH_LEN)

# ── 색상 ──────────────────────────────────────────────────────────────────────
C_RMS    = (50,  200, 255)
C_SCORE  = (200, 150, 255)
C_THR    = (0,    80, 255)
C_MOTION = (0,   255, 120)
C_SKIP   = (0,   160, 255)
C_ZERO   = (70,   70,  70)

frame_idx  = 0
paused     = False
last_frame = None
last_stab  = None
last_w_box = None
last_motion = False
last_rms   = 0.0


def draw_panel(rms: float, skip: bool, motion: bool, score: float, n_det: int) -> np.ndarray:
    panel = np.zeros((PANEL_H, args.display_w, 3), dtype=np.uint8)
    panel[:] = (25, 25, 25)

    rms_col   = C_THR if rms > rms_thr else (200, 200, 200)
    mot_col   = C_MOTION if motion else (100, 100, 100)
    skip_col  = C_SKIP if skip else (100, 100, 100)

    items = [
        ("RMS (flow magnitude)",  f"{rms:.4f}  (thr:{rms_thr})",  rms_col),
        ("YOLO detections",       f"{n_det} weight(s)",            (200, 200, 200)),
        ("Box jump skip",         "SKIPPED" if skip else "OK",     skip_col),
        ("Motion / Score",        f"{'MOVING' if motion else 'STILL'}  {score:.2f}", mot_col),
    ]
    col_w = args.display_w // len(items)
    for i, (label, value, color) in enumerate(items):
        x = i * col_w + 10
        cv2.putText(panel, label, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)
        cv2.putText(panel, value, (x, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    for i in range(1, len(items)):
        cv2.line(panel, (i * col_w, 8), (i * col_w, PANEL_H - 8), (60, 60, 60), 1)
    return panel


def draw_graph(total_w: int) -> np.ndarray:
    g = np.zeros((GRAPH_H, total_w, 3), dtype=np.uint8)
    g[:] = (20, 20, 20)

    mid = GRAPH_H // 2
    max_rms = max(max(rms_hist), rms_thr * 2, 1.0)

    # 기준선 (0)
    cv2.line(g, (0, GRAPH_H - 15), (total_w, GRAPH_H - 15), C_ZERO, 1)
    # threshold 선
    thr_y = int(GRAPH_H - 15 - rms_thr / max_rms * (GRAPH_H - 20))
    thr_y = max(2, min(GRAPH_H - 2, thr_y))
    cv2.line(g, (0, thr_y), (total_w, thr_y), C_THR, 1)
    cv2.putText(g, f"thr={rms_thr}", (4, thr_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, C_THR, 1)

    def hist_to_pts(hist, max_val, invert=False):
        pts = []
        base = GRAPH_H - 15
        for i, v in enumerate(hist):
            x = int(i / GRAPH_LEN * total_w)
            if invert:
                y = int((GRAPH_H - 15) * (1.0 - min(v, 1.0)))
            else:
                y = int(base - v / max_val * (base - 4))
            pts.append((x, max(2, min(GRAPH_H - 2, y))))
        return pts

    def draw_line(pts, color):
        for i in range(1, len(pts)):
            cv2.line(g, pts[i-1], pts[i], color, 1)

    draw_line(hist_to_pts(rms_hist,   max_rms),         C_RMS)
    draw_line(hist_to_pts(score_hist, 1.0, invert=True), C_SCORE)

    legend = [
        (f"RMS (max {max_rms:.1f})", C_RMS),
        ("motion score",             C_SCORE),
        (f"threshold={rms_thr}",     C_THR),
    ]
    for i, (label, color) in enumerate(legend):
        cv2.putText(g, label, (total_w - 200, 14 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    for sec in range(0, int(GRAPH_LEN / fps) + 1, 2):
        x = int(sec * fps / GRAPH_LEN * total_w)
        cv2.line(g, (x, GRAPH_H - 14), (x, GRAPH_H - 6), (60, 60, 60), 1)
        cv2.putText(g, f"{sec}s", (x + 2, GRAPH_H - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 80, 80), 1)
    return g


def process_frame(frame):
    global last_stab, last_w_box, last_motion, last_rms

    stabilized = stab.stabilize(frame)
    last_stab  = stabilized

    dets    = det.detect(stabilized)
    weights = [d for d in dets if d.class_id == CLASS_POT_WEIGHT and d.confidence >= 0.25]
    w_box   = None
    if weights:
        best  = max(weights, key=lambda d: d.confidence)
        w_box = (int(best.x1), int(best.y1), int(best.x2), int(best.y2))
    last_w_box = w_box

    motion, rms  = oflow.update(stabilized, w_box)
    last_motion  = motion
    last_rms     = rms

    rms_hist.append(rms)
    score_hist.append(oflow.score)
    return len(weights)


# ── 메인 루프 ─────────────────────────────────────────────────────────────────
n_det = 0
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("[End of video]")
            break
        last_frame = frame.copy()
        n_det = process_frame(frame)
        frame_idx += 1

    if last_frame is None:
        if cv2.waitKey(10) & 0xFF in (ord('q'), 27):
            break
        continue

    # ── 좌: 원본 + bbox ───────────────────────────────────────────────────
    vis_orig = cv2.resize(last_frame, (half_w, disp_h))
    if last_w_box:
        x1, y1, x2, y2 = last_w_box
        sx1 = int(x1 * scale); sy1 = int(y1 * scale)
        sx2 = int(x2 * scale); sy2 = int(y2 * scale)
        box_col = C_THR if last_motion else (0, 255, 255)
        cv2.rectangle(vis_orig, (sx1, sy1), (sx2, sy2), box_col, 2)
        cv2.putText(vis_orig, f"RMS:{last_rms:.3f}", (sx1, sy1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_col, 1)
    status_col = C_MOTION if last_motion else (150, 150, 150)
    status_txt = "MOVING" if last_motion else "STILL"
    cv2.putText(vis_orig, "Original + YOLO", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
    cv2.putText(vis_orig, f"{status_txt}  score:{oflow.score:.2f}", (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_col, 1)
    cv2.putText(vis_orig, f"frame {frame_idx}", (10, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

    # ── 우: 안정화 프레임 ─────────────────────────────────────────────────
    vis_stab = cv2.resize(last_stab, (half_w, disp_h))
    cv2.putText(vis_stab, "Stabilized (Phase 1)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
    skip_txt = "SKIP (box jumped)" if oflow.last_skipped else ""
    cv2.putText(vis_stab, skip_txt, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_SKIP, 1)

    # ── 중앙: 딸랑이 ROI 확대 + flow 오버레이 ────────────────────────────
    roi_panel = np.zeros((disp_h, ROI_W, 3), dtype=np.uint8)
    roi_panel[:] = (30, 30, 30)

    if last_w_box and oflow.last_flow is not None:
        x1, y1, x2, y2 = oflow.last_roi_box or last_w_box
        roi_crop = last_stab[y1:y2, x1:x2]
        if roi_crop.size > 0:
            flow_bgr = OpticalFlowDetector.flow_to_bgr(oflow.last_flow)
            rh, rw   = roi_crop.shape[:2]
            # 확대 비율 계산
            rs = min(ROI_W / max(rw, 1), (disp_h - 60) / max(rh, 1))
            nw, nh   = max(1, int(rw * rs)), max(1, int(rh * rs))
            ox = (ROI_W - nw) // 2
            oy = (disp_h - nh - 60) // 2 + 40

            roi_resized  = cv2.resize(roi_crop,  (nw, nh))
            flow_resized = cv2.resize(flow_bgr,  (nw, nh))

            # 원본 ROI + flow 50% 블렌드
            blended = cv2.addWeighted(roi_resized, 0.5, flow_resized, 0.5, 0)
            roi_panel[oy:oy+nh, ox:ox+nw] = blended

            # 박스 테두리
            col = C_THR if last_motion else (0, 255, 255)
            cv2.rectangle(roi_panel, (ox, oy), (ox+nw, oy+nh), col, 1)

    cv2.putText(roi_panel, "ROI + Flow", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(roi_panel, f"RMS {last_rms:.3f}", (5, disp_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_RMS, 1)
    cv2.putText(roi_panel, f"thr {rms_thr}", (5, disp_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_THR, 1)

    # ── 조합 ──────────────────────────────────────────────────────────────
    video_row = np.hstack([vis_orig, roi_panel, vis_stab])
    panel     = draw_panel(last_rms, oflow.last_skipped, last_motion, oflow.score, n_det)
    graph     = draw_graph(args.display_w)
    display   = np.vstack([video_row, panel, graph])

    cv2.imshow("Phase 2 Optical Flow  [space=pause  q=quit]", display)

    key = cv2.waitKey(1 if not paused else 30) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key == ord(' '):
        paused = not paused
        print(f"  {'PAUSED' if paused else 'RESUMED'}  frame={frame_idx}  RMS={last_rms:.4f}")
    elif paused and key == 83:  # → 방향키
        ret, frame = cap.read()
        if ret:
            last_frame = frame.copy()
            n_det = process_frame(frame)
            frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print(f"[Done] {frame_idx} frames")
