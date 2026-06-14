"""
압력밥솥 타이머 시스템 - 실제 파이프라인 다이어그램 생성
실행: uv run python docs/pipeline_diagram.py
출력: docs/pipeline.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import os

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(20, 14), dpi=110)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')
fig.patch.set_facecolor('#0f1115')
ax.set_facecolor('#0f1115')

# ===== 색상 팔레트 =====
COL_INPUT   = '#2c3e50'
COL_PHASE1  = '#16a085'
COL_YOLO    = '#8e44ad'
COL_MATCH   = '#d35400'
COL_PHASE2  = '#2980b9'
COL_STATE   = '#c0392b'
COL_UI      = '#7f8c8d'
COL_TEXT    = '#ecf0f1'
COL_ARROW   = '#bdc3c7'
COL_HILITE  = '#f1c40f'

def box(x, y, w, h, color, title, lines, edge='white', title_size=11, body_size=9):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.6",
                       linewidth=1.5, edgecolor=edge, facecolor=color, alpha=0.92)
    ax.add_patch(p)
    ax.text(x + w/2, y + h - 1.2, title, ha='center', va='top',
            fontsize=title_size, fontweight='bold', color='white')
    for i, ln in enumerate(lines):
        ax.text(x + 0.7, y + h - 2.8 - i*1.25, ln, ha='left', va='top',
                fontsize=body_size, color=COL_TEXT, family='monospace')

def arrow(x1, y1, x2, y2, color=COL_ARROW, style='-|>', mut=18, lw=2.0, label=None, lx=None, ly=None):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=mut, color=color, linewidth=lw,
                        connectionstyle="arc3,rad=0")
    ax.add_patch(a)
    if label:
        ax.text(lx if lx is not None else (x1+x2)/2, ly if ly is not None else (y1+y2)/2,
                label, ha='center', va='center', fontsize=8, color=COL_HILITE,
                bbox=dict(boxstyle='round,pad=0.25', fc='#0f1115', ec=COL_HILITE, lw=0.6))

# ===== 제목 =====
ax.text(50, 97.5, '압력밥솥 타이머 시스템 — 실제 파이프라인',
        ha='center', fontsize=20, fontweight='bold', color='white')
ax.text(50, 95.2, 'Phase 1 (Stabilizer)  →  YOLO-seg (ROI crop)  →  Matching  →  Phase 2 (Optical Flow)  →  State Machine  →  UI',
        ha='center', fontsize=10.5, color='#95a5a6', style='italic')

# ===== ① INPUT =====
box(2, 80, 18, 11, COL_INPUT, '① Video Source',
    ['sources/video_source.py',
     '',
     '· 카메라 / 파일 / RTSP',
     '· Win: CAP_DSHOW 강제',
     '· target_fps 자동 산정',
     '   cam=1~2 → 15fps',
     '   cam=3~6 → 10fps',
     '   cam=7+  →  8fps'])

# ===== ② PHASE 1 STABILIZER =====
box(24, 80, 20, 11, COL_PHASE1, '② Stabilizer (Phase 1)',
    ['core/stabilizer.py · source별',
     '',
     '· Grid LK 특징점 추적',
     '· RANSAC 이상치 제거',
     '· EMA warpAffine 보정',
     '· 카메라 흔들림 → 0',
     '',
     '→ stabilized frame'])

# ===== ③ YOLO =====
box(48, 80, 22, 11, COL_YOLO, '③ YOLO-seg 추론',
    ['core/detector.py  ·  models/pot_seg.pt',
     '',
     '· ROI 합집합 + margin=50 으로',
     '  프레임 crop 후 1회 추론',
     '· 클래스: empty_burner / pot_body / pot_weight',
     '· conf:  body ≥ 0.30,  weight ≥ 0.25',
     '· bbox + segmentation mask_xy 반환',
     '· 좌표를 stabilized 전체로 환원'])

# ===== ④ MATCHING =====
box(74, 80, 24, 11, COL_MATCH, '④ Burner Matching',
    ['frame_processor.detect_and_update()',
     '',
     '1) ROI 중심거리 → pot_body 매칭',
     '   (ROI 영역 + 20% margin 안)',
     '2) body bbox 안의 weight 후보 수집',
     '3) |w.cx − body.cx| 그리디 독점 배정',
     '   (한 weight = 한 화구)',
     '4) body 소실 시 TTL 15프레임 유지'])

# 단계 1 → 단계 2 → 단계 3 → 단계 4 가로 화살표
for x in [20, 44, 70]:
    arrow(x, 85.5, x + 4, 85.5)

# ===== 화구별 분기 표현 =====
ax.text(50, 76, '── 화구별 독립 처리 (per-burner) ──',
        ha='center', fontsize=11, color=COL_HILITE, fontweight='bold')

# ===== ⑤ PHASE 2 — OPTICAL FLOW =====
box(6, 50, 42, 24, COL_PHASE2, '⑤ Optical Flow Detector (Phase 2)',
    ['core/optical_flow.py  ·  화구별 1개 인스턴스',
     '',
     '입력 :  stabilized frame  +  weight bbox  +  mask_xy',
     '',
     '· crop 위치 : bbox center EMA  (pos_alpha=0.3)',
     '· Farneback dense optical flow 계산',
     '· RMS :  mask 폴리곤 내부 픽셀만',
     '          (mask 없는 프레임은 bbox 전체 fallback)',
     '· 크기 정규화 :  norm_rms = raw × ref_diag / bbox_diag',
     '          (ref_diag = 40px)',
     '· EMA 스무딩 :  alpha = 0.35',
     '· Window 투표 :  최근 25프레임 중 14프레임 ≥ threshold',
     '         →  진동 확정 (STEAMING)',
     '',
     'threshold:  rms_threshold = 0.5  (config)',
     '',
     '비활성:  Phase 3 (IIR bandpass 주파수 분석)'],
    body_size=8.5)

# ===== ⑥ STATE MACHINE =====
box(52, 50, 46, 24, COL_STATE, '⑥ Burner State Machine',
    ['core/state_machine.py  ·  7-state FSM + 타이머 잠금',
     '',
     '  EMPTY            ─pot감지→ POT_IDLE',
     '  POT_IDLE         ─진동확정→ POT_STEAMING_FIRST  ★잠금',
     '  STEAMING_FIRST   ─타이머완료→ DONE_FIRST  (12분)',
     '  DONE_FIRST       ─done_first_timeout→ WAIT_SECOND',
     '  WAIT_SECOND      ─진동확정→ POT_STEAMING_SECOND ★잠금',
     '  STEAMING_SECOND  ─타이머완료→ DONE_SECOND  (5분)',
     '  DONE_SECOND      ─pot이탈→ EMPTY',
     '',
     '★잠금 :  타이머 중에는 감지 결과로 자동 전환 X',
     '          수동 R(리셋) / S(강제시작)만 허용',
     '',
     'pot_absent_threshold = 30 프레임 → EMPTY 전환',
     'STATE_COLORS = UI 카드 색상 매핑 7종'],
    body_size=8.5)

# ===== 화살표 ④ → ⑤ ⑥ =====
arrow(85, 80, 70, 74.5)
arrow(85, 80, 30, 74.5)
arrow(30, 50, 70, 50, color=COL_HILITE, label='detections / vibration', ly=51.2)
arrow(70, 60, 52, 60, color=COL_HILITE, style='<|-', lw=1.5)
ax.text(61, 58.5, 'reset() on state-transition', ha='center', fontsize=8,
        color=COL_HILITE, style='italic')

# ===== ⑦ UI =====
box(6, 18, 88, 26, COL_UI, '⑦ UI Display  ·  ui/ui_display.py  (pygame)',
    ['',
     '┌───────────────────────────────────────────┬────────────────────────────┐',
     '│  좌측: 카메라 영상 오버레이                          │  우측 패널: 화구 카드 그리드        │',
     '│   · YOLO bbox (pot_body / pot_weight)             │   · 화구 ID · 남은 시간 · 상태색      │',
     '│   · segmentation mask (M 키 토글)                  │   · ⟳ 리셋(1s 길게)  · ▶ 강제시작     │',
     '│   · RMS 수치 (current_angle = smoothed RMS)        │   · ⚙ 설정 · Mask · 카메라 전환       │',
     '│   · F 키 → 인앱 캘리브레이션 (드래그로 ROI 지정)     │   · 1~9,0 키로 화구 직접 선택          │',
     '└───────────────────────────────────────────┴────────────────────────────┘',
     '',
     '핫 리로드 :  설정 저장 시 BurnerRegistry / FrameProcessor 재구성  (재시작 불필요)',
     '루프      :  pygame events → processor.read_frames() → processor.detect_and_update() → display.render()',
     '              ↑ frame_interval = 1/target_fps',
     ''],
    body_size=9)

arrow(30, 50, 30, 44, color=COL_ARROW)
arrow(70, 50, 70, 44, color=COL_ARROW)

# ===== ⑧ CONFIG =====
box(2, 2, 96, 13, '#34495e', '⑧ Config & 데이터 흐름',
    ['config/store_config.json   (핫 리로드 가능)',
     '',
     '  sources[]  → VideoSource 생성  ·  source-level stabilizer / optical_flow override 허용',
     '  burners[]  → BurnerStateMachine 생성  ·  roi, countdown_first/second, done_first_timeout, pot_absent_threshold',
     '  model      → weights = models/pot_seg.pt  ·  confidence = 0.5',
     '  optical_flow → rms_threshold, rms_ema_alpha=0.35, window_frames=25, trigger_frames=14',
     '',
     '합성 우선순위:  global  <  source.optical_flow  <  burner.optical_flow         (해상도/모델 혼합 환경 지원)'],
    body_size=8.5)

# ===== 범례 =====
legend_items = [
    ('Input',      COL_INPUT),
    ('Phase 1',    COL_PHASE1),
    ('YOLO',       COL_YOLO),
    ('Matching',   COL_MATCH),
    ('Phase 2',    COL_PHASE2),
    ('State FSM',  COL_STATE),
    ('UI',         COL_UI),
]
for i, (name, c) in enumerate(legend_items):
    cx = 4 + i * 13
    p = FancyBboxPatch((cx, 0.2), 2.2, 1.2, boxstyle="round,pad=0.1",
                       facecolor=c, edgecolor='white', linewidth=0.7)
    ax.add_patch(p)
    ax.text(cx + 2.6, 0.8, name, ha='left', va='center', fontsize=8.5, color='white')

out = os.path.join(os.path.dirname(__file__), 'pipeline.png')
plt.savefig(out, dpi=130, facecolor='#0f1115', bbox_inches='tight', pad_inches=0.3)
print(f'saved: {out}')
