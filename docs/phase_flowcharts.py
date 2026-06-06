"""
Phase 1 (Stabilizer) + Phase 2 (OpticalFlow) 플로우차트
실행: uv run python docs/phase_flowcharts.py
출력: docs/phase1.png, docs/phase2.png
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BG       = '#0f1115'
TEXT     = '#ecf0f1'
HILITE   = '#f1c40f'
ARR      = '#bdc3c7'

# 노드 색상
C_IO     = '#34495e'   # 입출력
C_PROC   = '#2980b9'   # 일반 처리
C_DEC    = '#8e44ad'   # 분기/판단
C_FAIL   = '#c0392b'   # 실패/리셋 경로
C_OK     = '#16a085'   # 성공 경로
C_KEY    = '#d35400'   # 핵심 단계


def rect(ax, x, y, w, h, color, title, lines=None, fs_t=10.5, fs_b=9, edge='white', alpha=0.92):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.3,rounding_size=0.5",
                       linewidth=1.4, edgecolor=edge, facecolor=color, alpha=alpha)
    ax.add_patch(p)
    if lines:
        ax.text(x, y + h/2 - 0.9, title, ha='center', va='top',
                fontsize=fs_t, fontweight='bold', color='white')
        for i, ln in enumerate(lines):
            ax.text(x, y + h/2 - 2.0 - i*1.15, ln, ha='center', va='top',
                    fontsize=fs_b, color=TEXT, family='monospace')
    else:
        ax.text(x, y, title, ha='center', va='center',
                fontsize=fs_t, fontweight='bold', color='white')


def diamond(ax, x, y, w, h, color, title, fs=10):
    pts = [(x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)]
    poly = Polygon(pts, closed=True, facecolor=color, edgecolor='white',
                   linewidth=1.4, alpha=0.92)
    ax.add_patch(poly)
    ax.text(x, y, title, ha='center', va='center', fontsize=fs,
            fontweight='bold', color='white')


def arr(ax, x1, y1, x2, y2, color=ARR, label=None, lx=None, ly=None,
        label_color=None, lw=2.0, rad=0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                        mutation_scale=18, color=color, linewidth=lw,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if label:
        ax.text(lx if lx is not None else (x1+x2)/2,
                ly if ly is not None else (y1+y2)/2,
                label, ha='center', va='center', fontsize=8.5,
                color=label_color or HILITE,
                bbox=dict(boxstyle='round,pad=0.25', fc=BG,
                          ec=label_color or HILITE, lw=0.6))


# ========================================================================
#                              PHASE 1
# ========================================================================
fig, ax = plt.subplots(figsize=(14, 19), dpi=110)
ax.set_xlim(0, 100); ax.set_ylim(0, 130); ax.axis('off')
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

ax.text(50, 126, 'Phase 1 — Stabilizer',
        ha='center', fontsize=22, fontweight='bold', color='white')
ax.text(50, 123, 'core/stabilizer.py · 카메라 흔들림 제거 (frame-to-frame, drift-free)',
        ha='center', fontsize=11, color='#95a5a6', style='italic')

# 입력
rect(ax, 50, 117, 30, 4, C_IO, 'INPUT: frame (BGR)')

# Step 0: gray 변환
rect(ax, 50, 110, 36, 4.5, C_PROC, 'cv2.cvtColor → gray')
arr(ax, 50, 115, 50, 112.5)

# 분기 1: 첫 프레임?
diamond(ax, 50, 102, 28, 6, C_DEC, 'prev_gray is None ?')
arr(ax, 50, 107.5, 50, 105)

# 첫 프레임 → 특징점 저장 → return
rect(ax, 80, 102, 24, 5.5, C_OK,
     'goodFeaturesToTrack\nprev_gray ← gray\nreturn frame',
     lines=['goodFeaturesToTrack(qLvl=0.03)',
            '_prev_pts 저장',
            'return frame (보정 X)'], fs_b=8)
arr(ax, 64, 102, 68, 102, label='YES (첫 프레임)', lx=66, ly=104, label_color=C_OK)

# Step 1: LK 추적
rect(ax, 50, 92, 50, 7, C_KEY, 'Step 1 · LK Optical Flow',
     lines=['cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts)',
            'good_prev, good_curr  =  status == 1 인 점만',
            'last_n_features = len(good_prev)'], fs_b=8.5)
arr(ax, 50, 99, 50, 95.5, label='NO', lx=53, ly=97.5)

# 분기 2: 특징점 부족?
diamond(ax, 50, 83, 32, 6, C_DEC, 'good_prev < min_inliers(6) ?')
arr(ax, 50, 88.5, 50, 86)

# 추적 실패 경로
rect(ax, 86, 83, 22, 5.5, C_FAIL,
     '추적 실패 → 재시도',
     lines=['특징점 재생성',
            'return frame (보정 X)'], fs_b=8)
arr(ax, 66, 83, 75, 83, label='YES', lx=70, ly=84.5, label_color=C_FAIL)

# Step 2: RANSAC
rect(ax, 50, 74, 60, 7, C_KEY, 'Step 2 · RANSAC affine 추정',
     lines=['cv2.estimateAffinePartial2D(method=RANSAC, thr=3.0px)',
            '배경 인라이어(공통 평행이동) 만 선별 → 움직이는 물체는 제외',
            'last_n_inliers = inlier_mask.sum()'], fs_b=8.5)
arr(ax, 50, 80, 50, 77.5, label='NO', lx=53, ly=78.7)

# 분기 3: RANSAC 실패?
diamond(ax, 50, 65, 36, 6, C_DEC, 'transform None  or  inliers < 6 ?')
arr(ax, 50, 70.5, 50, 68)

rect(ax, 86, 65, 22, 5.5, C_FAIL,
     'RANSAC 실패',
     lines=['특징점 재생성',
            'return frame (보정 X)'], fs_b=8)
arr(ax, 68, 65, 75, 65, label='YES', lx=71, ly=66.5, label_color=C_FAIL)

# Step 3: EMA 보정
rect(ax, 50, 55, 60, 8, C_KEY, 'Step 3 · EMA 스무딩 + warpAffine',
     lines=['raw_dx, raw_dy = transform[:, 2]',
            'smooth_dx = α·raw_dx + (1−α)·smooth_dx     (α=0.3)',
            'correction = [[1, 0, −smooth_dx], [0, 1, −smooth_dy]]',
            'stabilized = cv2.warpAffine(frame, correction)'], fs_b=8.5)
arr(ax, 50, 62, 50, 59, label='NO', lx=53, ly=60.5)

# 갱신
rect(ax, 50, 44, 50, 5.5, C_PROC, '다음 프레임 준비',
     lines=['prev_gray ← gray',
            'prev_pts ← _detect_features(gray)  (매 프레임 재검출)'], fs_b=8.5)
arr(ax, 50, 51, 50, 46.7)

# 출력
rect(ax, 50, 36, 30, 4.5, C_IO, 'OUTPUT: stabilized frame')
arr(ax, 50, 41.2, 50, 38.2)

# 사이드: 핵심 설계 포인트
rect(ax, 18, 22, 32, 28, '#1c1f26', '설계 포인트',
     lines=['',
            '· frame-to-frame 보정',
            '   → 누적 drift 0',
            '',
            '· EMA(α=0.3)',
            '   → 급격한 튐 억제',
            '   → 저주파 이동은 통과',
            '',
            '· Shi-Tomasi 우선,',
            '   특징점 < 6 → Grid fallback',
            '',
            '· 실패 시 보정 안 함',
            '   (왜곡보다는 무보정이 안전)'], fs_b=8.5, edge='#7f8c8d')

# 사이드: 모니터링 속성
rect(ax, 76, 22, 32, 28, '#1c1f26', 'last_* 모니터링 속성',
     lines=['',
            'last_n_features',
            '   추적된 특징점 수',
            '',
            'last_n_inliers',
            '   RANSAC 인라이어 수',
            '',
            'last_raw_dx / dy',
            '   원본 보정량',
            '',
            'last_smooth_dx / dy',
            '   EMA 후 적용된 보정량'], fs_b=8.5, edge='#7f8c8d')

out1 = os.path.join(os.path.dirname(__file__), 'phase1.png')
plt.savefig(out1, dpi=130, facecolor=BG, bbox_inches='tight', pad_inches=0.3)
plt.close(fig)
print(f'saved: {out1}')


# ========================================================================
#                              PHASE 2
# ========================================================================
fig, ax = plt.subplots(figsize=(16, 22), dpi=110)
ax.set_xlim(0, 100); ax.set_ylim(0, 150); ax.axis('off')
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

ax.text(50, 146, 'Phase 2 — Optical Flow Detector',
        ha='center', fontsize=22, fontweight='bold', color='white')
ax.text(50, 143, 'core/optical_flow.py · 화구별 1개 · 딸랑이 진동 판정',
        ha='center', fontsize=11, color='#95a5a6', style='italic')

# INPUT
rect(ax, 50, 137, 50, 5, C_IO,
     'INPUT',
     lines=['stabilized frame  +  weight bbox  +  mask_xy (폴리곤)'], fs_b=9)

# 분기: w_box is None?
diamond(ax, 50, 129, 26, 5.5, C_DEC, 'w_box is None ?')
arr(ax, 50, 134.5, 50, 131.75)

# missing 처리
rect(ax, 82, 129, 30, 6, C_FAIL,
     'missing_streak += 1',
     lines=['streak ≥ 3 → cv_hist clear',
            'EMA centroid는 유지',
            'return  (check(), 0.0)'], fs_b=8.5)
arr(ax, 63, 129, 67, 129, label='YES', lx=65, ly=130.5, label_color=C_FAIL)

# 분기: bbox tiny?
diamond(ax, 50, 121, 28, 5.5, C_DEC, 'bbox < 8 × 8 ?')
arr(ax, 50, 126.25, 50, 123.75, label='NO', lx=53, ly=125)

rect(ax, 82, 121, 30, 5, C_FAIL,
     'tiny_box',
     lines=['_handle_gap("tiny_box")',
            'return'], fs_b=8.5)
arr(ax, 64, 121, 67, 121, label='YES', lx=65.5, ly=122.5, label_color=C_FAIL)

# Step 1: box jump
rect(ax, 50, 112, 64, 7, C_KEY, 'Step 1 · Box Jump 감지',
     lines=['jump_px = ‖curr_center − prev_raw_box_center‖',
            'jump_triggered =  (jump_px / bbox_diag) > 0.5',
            '→ 옆 화구 weight로 추적이 튄 경우 감지'], fs_b=8.5)
arr(ax, 50, 118.25, 50, 115.5, label='NO', lx=53, ly=116.75)

# 분기: jump?
diamond(ax, 50, 103, 22, 5.5, C_DEC, 'jump ?')
arr(ax, 50, 108.5, 50, 105.75)

rect(ax, 82, 103, 30, 6.5, C_FAIL,
     '새 추적 시작',
     lines=['centroid EMA = None',
            'ema_rms = 0.0',
            'cv_hist clear',
            'return'], fs_b=8.5)
arr(ax, 61, 103, 67, 103, label='YES', lx=64, ly=104.5, label_color=C_FAIL)

# Step 2: centroid EMA
rect(ax, 50, 93, 64, 8, C_KEY, 'Step 2 · Crop 위치 안정화 (centroid EMA)',
     lines=['raw_cx, raw_cy = bbox center',
            'ema_cx = α·raw_cx + (1−α)·ema_cx     (α = pos_alpha = 0.3)',
            '→ YOLO bbox jitter(±1~2px) 흡수',
            'crop = frame[ema_center ± half_box]'], fs_b=8.5)
arr(ax, 50, 100.25, 50, 97, label='NO', lx=53, ly=98.5)

# 분기: 첫 ROI?
diamond(ax, 50, 84, 28, 5.5, C_DEC, 'prev_roi_gray None ?')
arr(ax, 50, 89, 50, 86.75)

rect(ax, 82, 84, 30, 5, C_PROC,
     'new_track',
     lines=['prev_roi_gray ← roi_gray',
            'return  (cv_hist 유지)'], fs_b=8.5)
arr(ax, 64, 84, 67, 84, label='YES', lx=65.5, ly=85.5, label_color=C_OK)

# Step 3: Farneback
rect(ax, 50, 74, 70, 7, C_KEY, 'Step 3 · Farneback Dense Optical Flow',
     lines=['flow = cv2.calcOpticalFlowFarneback(prev_roi_gray, roi_gray, ...)',
            '   pyr_scale=0.5  ·  levels=3  ·  winsize=15  ·  iter=3',
            '→ ROI 모든 픽셀의 (dx, dy) 이동벡터'], fs_b=8.5)
arr(ax, 50, 81.25, 50, 77.5, label='NO', lx=53, ly=79.5)

# Step 4: Residual RMS
rect(ax, 50, 64, 75, 8, C_KEY, 'Step 4 · Residual RMS (mean flow 차감 → deformation만 측정)',
     lines=['mask 있음: mask 폴리곤 내부 픽셀만 사용',
            '   mean_flow = mask 내부 평균  (=카메라 떨림 + 위치 이동)',
            '   rms = √mean((fx − mean_fx)² + (fy − mean_fy)²)     ← 차감 후 잔차',
            'mask 없음: bbox 전체 픽셀로 동일 방식 fallback'], fs_b=8.5)
arr(ax, 50, 70.5, 50, 68, label='이미지 흐름이 있을 때만', lx=63, ly=69, label_color=C_OK)

# Step 5: 정규화 + EMA
rect(ax, 50, 53, 75, 8, C_KEY, 'Step 5 · 크기 정규화 + RMS EMA',
     lines=['if normalize_rms:  norm_rms = rms × ref_diag / bbox_diag     (ref=40px)',
            '   → 해상도·줌이 달라도 threshold 동일하게 적용 가능',
            'ema_rms = α·norm_rms + (1−α)·ema_rms     (α = rms_ema_alpha)',
            'last_smoothed_rms = ema_rms     (UI 표시값)'], fs_b=8.5)
arr(ax, 50, 60, 50, 57)

# Step 6: window 투표
rect(ax, 50, 42, 75, 8.5, C_KEY, 'Step 6 · Window 투표',
     lines=['motion = ema_rms > rms_threshold     (0.5)',
            'cv_hist.append(motion)     (deque, maxlen = window_frames = 25)',
            '',
            'vibrating = len(cv_hist) == 25  AND  sum(cv_hist) ≥ trigger_frames(12)',
            '→ "잠깐 흔들림" 이 아닌  "지속 흔들림"  확정'], fs_b=8.5)
arr(ax, 50, 49, 50, 46.25)

# OUTPUT
rect(ax, 50, 32, 60, 6, C_IO,
     'OUTPUT',
     lines=['(vibrating: bool, rms: float)   → FrameProcessor → State Machine'], fs_b=9)
arr(ax, 50, 37.75, 50, 35)

# 사이드: 파라미터
rect(ax, 16, 16, 30, 22, '#1c1f26', '핵심 파라미터',
     lines=['',
            'rms_threshold       0.5',
            'rms_ema_alpha       0.5',
            'pos_ema_alpha       0.3',
            'window_frames        25',
            'trigger_frames       12',
            'normalize_ref_diag  40 px',
            'max_box_jump_ratio  0.5',
            'missing_reset_frames  3'], fs_b=8.5, edge='#7f8c8d')

# 사이드: 5단 평활화
rect(ax, 50, 16, 32, 22, '#1c1f26', '5중 노이즈 제거',
     lines=['',
            '① centroid EMA',
            '   → bbox jitter',
            '',
            '② residual (mean 차감)',
            '   → 카메라 떨림',
            '',
            '③ mask 폴리곤',
            '   → 배경 희석',
            '',
            '④ RMS EMA',
            '   → 단일 프레임 스파이크',
            '',
            '⑤ Window 투표',
            '   → 일시적 흔들림'], fs_b=8.5, edge='#7f8c8d')

# 사이드: reset 트리거
rect(ax, 84, 16, 30, 22, '#1c1f26', 'reset() / clear 트리거',
     lines=['',
            '· box jump (jump>0.5×diag)',
            '· missing 3프레임 연속',
            '· tiny box (<8×8)',
            '',
            '· 외부 호출 (state 전환):',
            '   POT_IDLE → STEAMING',
            '   STEAMING → DONE',
            '   DONE_FIRST → WAIT_2',
            '   EMPTY 진입',
            '',
            '→ cv_hist / ema_rms 초기화'], fs_b=8.5, edge='#7f8c8d')

out2 = os.path.join(os.path.dirname(__file__), 'phase2.png')
plt.savefig(out2, dpi=130, facecolor=BG, bbox_inches='tight', pad_inches=0.3)
plt.close(fig)
print(f'saved: {out2}')
