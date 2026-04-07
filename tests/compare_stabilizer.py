"""
Phase 1 - Stabilizer 효과 비교 뷰어
=====================================

[적용한 것]
  Step 1. goodFeaturesToTrack
          - 영상에서 추적하기 좋은 코너 특징점을 최대 200개 검출 (Shi-Tomasi 알고리즘)
          - 노란 점으로 표시됨

  Step 2. calcOpticalFlowPyrLK + RANSAC
          - Lucas-Kanade 알고리즘으로 특징점이 다음 프레임에서 어디로 이동했는지 추적
          - RANSAC으로 "같은 방향으로 움직인 배경 점(인라이어)"만 선별
          - 딸랑이처럼 혼자 튀는 점은 아웃라이어로 제거됨
          - 선별된 배경 점들의 공통 이동량 = 카메라 흔들림 벡터

  Step 3. warpAffine
          - 카메라 흔들림 벡터를 역방향으로 적용해 프레임을 원위치로 보정
          - 누적 적용 → 항상 첫 프레임 기준으로 정렬됨

[화면 구성]
  좌: 원본 (노란 점 = 추적 중인 특징점)
  우: 안정화 결과 + 누적 보정량(dx, dy)
  하단 패널: 보정량 수치 / 특징점 수 / 인라이어 비율 / 상태
  그래프: dx(파랑) dy(민트) 인라이어비율(보라) 시계열

[보는 방법]
  - 그래프가 0 근처에 평평 → 카메라 거의 고정
  - 그래프가 점진적으로 drift → 카메라가 천천히 움직임
  - 그래프가 출렁임 → 카메라 진동 (환풍기 등) → Phase 1 효과 극대화
  - 인라이어 비율 80% 이하 → 화면에 큰 변화 있음 (연기, 사람 등)
  - 누적 보정량 > 5px → 주황색으로 강조

[키 조작]
  space : 일시정지 / 재개
  q/ESC : 종료
  → 방향키 : 일시정지 중 한 프레임 앞으로

실행:
  cd c:\\Solo\\Pressure_Chicken
  uv run python tests/compare_stabilizer.py
  uv run python tests/compare_stabilizer.py --video raw/Sample02.mp4
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
from core.stabilizer import Stabilizer

# ── 인자 파싱 ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--video",     default="raw/Sample01.mp4")
parser.add_argument("--display-w", type=int,   default=1600)
parser.add_argument("--graph-sec", type=float, default=10.0)
args = parser.parse_args()

# ── config 로드 ───────────────────────────────────────────────────────────────
with open("config/store_config.json", encoding="utf-8") as f:
    cfg = json.load(f)
stab_cfg = cfg.get("stabilizer", {})

# ── 영상 열기 ────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"[ERROR] Cannot open: {args.video}")
    sys.exit(1)

fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[Video] {args.video}  {vid_w}x{vid_h}  {total}f  {fps:.1f}fps")
print("  space: pause/resume  ->: step frame  q/ESC: quit")

# ── Stabilizer 초기화 ─────────────────────────────────────────────────────────
stab = Stabilizer(stab_cfg)

# ── 레이아웃 계산 ────────────────────────────────────────────────────────────
PANEL_H  = 80
GRAPH_H  = 100
half_w   = args.display_w // 2
scale    = half_w / vid_w
disp_h   = int(vid_h * scale)

GRAPH_LEN   = int(fps * args.graph_sec)
dx_hist:    deque[float] = deque([0.0] * GRAPH_LEN, maxlen=GRAPH_LEN)
dy_hist:    deque[float] = deque([0.0] * GRAPH_LEN, maxlen=GRAPH_LEN)
ratio_hist: deque[float] = deque([0.0] * GRAPH_LEN, maxlen=GRAPH_LEN)

# ── 색상 정의 ────────────────────────────────────────────────────────────────
C_ORIG  = (0,   180, 255)
C_STAB  = (0,   255, 120)
C_DX    = (50,  200, 255)
C_DY    = (50,  255, 150)
C_RATIO = (200, 150, 255)
C_WARN  = (0,    80, 255)
C_ZERO  = (80,   80,  80)

frame_idx = 0
paused    = False
last_orig = None
last_stab = None


def draw_panel(dx: float, dy: float, n_feat: int, n_inlier: int) -> np.ndarray:
    """수치 패널: 보정량 / 특징점 / 인라이어 비율 / 상태"""
    panel = np.zeros((PANEL_H, args.display_w, 3), dtype=np.uint8)
    panel[:] = (25, 25, 25)

    total_shift = (dx**2 + dy**2) ** 0.5
    ratio       = n_inlier / n_feat if n_feat > 0 else 0.0
    shift_col   = C_WARN if total_shift > 5 else (200, 200, 200)
    ratio_col   = (0, 200, 80) if ratio > 0.8 else C_WARN

    # cv2.putText는 한글 미지원 → 영문 사용
    items = [
        ("Cumul. Shift",  f"dx {dx:+.1f}px  dy {dy:+.1f}px  |{total_shift:.1f}|px", shift_col),
        ("Features",      f"{n_feat} tracked",                                         (200, 200, 200)),
        ("Inliers",       f"{n_inlier}/{n_feat}  ({ratio*100:.0f}%)",                  ratio_col),
        ("Status",        "SHAKE DETECTED" if total_shift > 5 else "STABLE",           shift_col),
    ]

    col_w = args.display_w // len(items)
    for i, (label, value, color) in enumerate(items):
        x = i * col_w + 10
        cv2.putText(panel, label, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
        cv2.putText(panel, value, (x, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

    for i in range(1, len(items)):
        cv2.line(panel, (i * col_w, 8), (i * col_w, PANEL_H - 8), (60, 60, 60), 1)

    return panel


def draw_graph(total_w: int) -> np.ndarray:
    """dx / dy / inlier ratio 시계열 그래프"""
    g = np.zeros((GRAPH_H, total_w, 3), dtype=np.uint8)
    g[:] = (20, 20, 20)

    mid = GRAPH_H // 2
    top = GRAPH_H - 10

    # 기준선
    cv2.line(g, (0, mid), (total_w, mid), C_ZERO, 1)
    cv2.putText(g, "0px", (4, mid - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_ZERO, 1)

    max_shift = max(max(abs(v) for v in dx_hist), max(abs(v) for v in dy_hist), 3.0)

    def hist_to_pts(hist, invert=False):
        pts = []
        for i, v in enumerate(hist):
            x = int(i / GRAPH_LEN * total_w)
            y = int(top * (1.0 - v)) if invert else int(mid - v / max_shift * (mid - 4))
            pts.append((x, max(2, min(GRAPH_H - 2, y))))
        return pts

    def draw_line(pts, color):
        for i in range(1, len(pts)):
            cv2.line(g, pts[i-1], pts[i], color, 1)

    draw_line(hist_to_pts(dx_hist),                C_DX)
    draw_line(hist_to_pts(dy_hist),                C_DY)
    draw_line(hist_to_pts(ratio_hist, invert=True), C_RATIO)

    # 범례 (영문)
    legend = [
        (f"raw dx (+-{max_shift:.1f}px)", C_DX),
        (f"raw dy (+-{max_shift:.1f}px)", C_DY),
        ("inlier ratio",                  C_RATIO),
    ]
    for i, (label, color) in enumerate(legend):
        cv2.putText(g, label, (total_w - 230, 16 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # x축 시간 눈금
    for sec in range(0, int(GRAPH_LEN / fps) + 1, 2):
        x = int(sec * fps / GRAPH_LEN * total_w)
        cv2.line(g, (x, GRAPH_H - 12), (x, GRAPH_H - 4), (60, 60, 60), 1)
        cv2.putText(g, f"{sec}s", (x + 2, GRAPH_H - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1)

    return g


# ── 메인 루프 ────────────────────────────────────────────────────────────────
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("[End of video]")
            break

        stabilized = stab.stabilize(frame)
        last_orig  = frame.copy()
        last_stab  = stabilized.copy()

        dx_hist.append(stab.last_raw_dx)
        dy_hist.append(stab.last_raw_dy)
        n_feat   = stab.last_n_features
        n_inlier = stab.last_n_inliers
        ratio_hist.append(n_inlier / n_feat if n_feat > 0 else 0.0)

        frame_idx += 1

    if last_orig is None:
        if cv2.waitKey(10) & 0xFF in (ord('q'), 27):
            break
        continue

    # ── 좌: 원본 + 격자점 ────────────────────────────────────────────────
    vis_orig = cv2.resize(last_orig, (half_w, disp_h))
    if stab._prev_pts is not None:
        for pt in stab._prev_pts:
            x, y = int(pt[0][0] * scale), int(pt[0][1] * scale)
            cv2.circle(vis_orig, (x, y), 3, (0, 255, 255), -1)
            cv2.circle(vis_orig, (x, y), 3, (0, 0, 0), 1)
    cv2.putText(vis_orig, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_ORIG, 2)
    cv2.putText(vis_orig, f"frame {frame_idx}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # ── 우: 안정화 결과 ──────────────────────────────────────────────────
    vis_stab  = cv2.resize(last_stab, (half_w, disp_h))
    raw_dx    = stab.last_raw_dx
    raw_dy    = stab.last_raw_dy
    smooth_dx = stab.last_smooth_dx
    smooth_dy = stab.last_smooth_dy
    total_sh  = (raw_dx**2 + raw_dy**2) ** 0.5
    shift_col = C_WARN if total_sh > 2 else (200, 200, 200)

    cv2.putText(vis_stab, "Stabilized (Phase 1 Grid)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_STAB, 2)
    cv2.putText(vis_stab, f"raw dx:{raw_dx:+.2f} dy:{raw_dy:+.2f}px",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, shift_col, 1)
    cv2.putText(vis_stab, f"smooth dx:{smooth_dx:+.2f} dy:{smooth_dy:+.2f}px",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # ── 조합 및 표시 ─────────────────────────────────────────────────────
    video_row = np.hstack([vis_orig, vis_stab])
    panel     = draw_panel(raw_dx, raw_dy, stab.last_n_features, stab.last_n_inliers)
    graph     = draw_graph(args.display_w)
    display   = np.vstack([video_row, panel, graph])

    cv2.imshow("Phase 1 Stabilizer  [space=pause  q=quit]", display)

    key = cv2.waitKey(1 if not paused else 30) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key == ord(' '):
        paused = not paused
        print(f"  {'PAUSED' if paused else 'RESUMED'}  frame={frame_idx}")
    elif paused and key == 83:  # → 방향키
        ret, frame = cap.read()
        if ret:
            stabilized = stab.stabilize(frame)
            last_orig  = frame.copy()
            last_stab  = stabilized.copy()
            frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print(f"[Done] {frame_idx} frames  last raw: dx={stab.last_raw_dx:+.2f}  dy={stab.last_raw_dy:+.2f}px")
