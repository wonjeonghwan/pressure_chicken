# 압력밥솥 타이머 시스템 (Pressure Cooker Timer)

## 프로젝트 개요

음식점 주방에서 여러 개의 압력밥솥을 동시에 운영할 때,
각 화구의 압력밥솥 딸랑이(추)가 움직이기 시작하면 자동으로 카운트다운 타이머를 시작하는
비전 기반 모니터링 시스템.

**인터넷 연결**: 학습 시 `yolov8n-seg.pt` 베이스 모델 최초 1회 다운로드만 필요.
이후 실행(운영)은 인터넷 없이 완전히 로컬에서 동작.

---

## 타이머 사이클 & 잠금 규칙

### 타이머 사이클
```
밥솥 올라옴 → 딸랑이 감지 → 초벌 12분 시작
초벌 완료 → DONE_FIRST(냉각, 진동 무시) → done_first_timeout(기본 10분) 경과 → 재벌 대기
재벌 대기(WAIT_SECOND, 진동 재활성) → 딸랑이 재감지 → 재벌 5분 시작
재벌 완료 → 경보
밥솥 이탈 (어느 단계든) → 전체 리셋 → 새 사이클
```

> **냉각 단계(DONE_FIRST)의 의미**: 초벌 완료 직후에는 잔여 진동이 남아 있어 곧바로
> "재벌 시작"으로 오인할 수 있다. 그래서 완료 후 `done_first_timeout` 동안은 진동을
> 무시하고(DONE_FIRST), 시간이 지나야 `WAIT_SECOND`로 넘어가 진동 감지를 재활성한다.

### 타이머 잠금
- 타이머 진행 중에는 카메라 감지 결과로 상태가 자동 변경되지 않음
- 딸랑이가 가려졌다가 다시 보여도 "또 시작"으로 오인하지 않음
- **수동 조작만** 상태를 바꿀 수 있음 (R키 초기화, S키/▶버튼 강제시작)

---

## 화구 상태 정의 (7개)

| 상태 | 설명 | 진동 감지 | UI 색상 (RGB) |
|------|------|-----------|---------------|
| `EMPTY` | 빈 화구 (밥솥 미감지) | — | 회색 `(80,80,80)` |
| `POT_IDLE` | 밥솥 감지, 대기 중 | 활성 | 브랜드 옐로우 `(255,192,0)` |
| `POT_STEAMING_FIRST` | 초벌 12분 타이머 진행 중 | 🔒 잠금 | 초록 `(60,180,60)` |
| `DONE_FIRST` | 초벌 완료, 냉각 중 (진동 무시, timeout 후 자동 전환) | ❌ 무시 | 오렌지 `(255,140,0)` |
| `WAIT_SECOND` | 재벌 대기 (진동 재활성) | 활성 | 파랑 `(100,160,220)` |
| `POT_STEAMING_SECOND` | 재벌 5분 타이머 진행 중 | 🔒 잠금 | 진초록 `(30,130,30)` |
| `DONE_SECOND` | 재벌 완료, 경보 (pot 이탈 시만 EMPTY) | — | 빨강 점멸 `(220,40,40)` |

**상태 전환 규칙**
```
EMPTY → POT_IDLE              : pot_body 감지
POT_IDLE → STEAMING_FIRST     : 딸랑이 진동 확정
STEAMING_FIRST → DONE_FIRST   : 초벌 타이머 완료
DONE_FIRST → WAIT_SECOND      : done_first_timeout 경과 + pot 존재
DONE_FIRST → EMPTY            : pot 이탈
WAIT_SECOND → STEAMING_SECOND : 딸랑이 진동 확정
WAIT_SECOND → EMPTY           : pot 이탈
STEAMING_SECOND → DONE_SECOND : 재벌 타이머 완료
DONE_SECOND → EMPTY           : pot 이탈 (최종 상태)
```

> **pot 이탈 판정(`pot_absent_threshold`)**: 완료 계열 상태(DONE_FIRST / WAIT_SECOND /
> DONE_SECOND)에서 밥솥을 **연속 N프레임(기본 30, ~30fps 기준 약 1초)** 못 봐야
> "치워짐 → EMPTY"로 확정한다. 김·손·행주에 순간 가려도 타이머를 날리지 않기 위한
> debounce 값. EMPTY 전환 시 내부 타이머 상태도 함께 초기화(`_reset_timer`)된다.

---

## YOLO 학습 클래스 (3가지)

```yaml
# dataset/data_aug.yaml (2026-07 재학습부터 — body 세그멘테이션 전환 + vent 추가)
nc: 3
names:
  0: body        # 밥솥 몸체 (기준점 역할, 세그멘테이션)
  1: pot_weight   # 딸랑이 (추)
  2: vent         # 증기 배출구 — 학습만 시키고 실사용 안 함(weight와 혼동 방지 목적)
```

> empty_burner 클래스는 2026-07 재학습부터 데이터셋에서 빠졌다(운영 코드에서도 원래
> 미사용 — EMPTY 상태는 pot_body 부재로 판정하지, 별도 클래스로 판정하지 않는다).
> `core/detector.py`의 `CLASS_POT_BODY`/`CLASS_POT_WEIGHT` 상수가 이 순서와 반드시
> 일치해야 하며, 재학습 후 모델 교체와 상수 변경은 항상 동시에 이루어져야 한다.

현재 모델: `models/pot_seg.pt` (yolov8n-seg 기반 segmentation 모델)

---

## 현재 확정된 감지 스택

```
영상 프레임 입력
  ↓
Phase 1: Stabilizer (core/stabilizer.py)
  - LK 특징점 추적 + RANSAC + EMA warpAffine
  - 카메라 흔들림 제거
  ↓
YOLO-seg 배치 추론 (ROI 합집합 crop 1회 호출, 동적 margin=max(30, min(50, ROI_short×0.1)))
  ↓
body/weight 매칭 (ROI 중심거리 기반 그리디)
  ↓
Phase 2: OpticalFlowDetector (core/optical_flow.py)
  - Farneback dense flow
  - crop 위치: bbox center EMA (pos_alpha=0.3) — mask 유무 무관하게 항상 bbox center 기준
  - RMS 계산: mask_xy 폴리곤 내부 픽셀만 (mask 없는 프레임은 bbox 전체 fallback)
  - 크기 정규화: norm_rms = raw_rms × (ref_diag / bbox_diag) ^ gamma  (normalize_rms=true 시)
    → 해상도·줌 변화에 무관하게 threshold 동일 적용 가능. ref_diag=40px 기준.
    → gamma(기본 1.0=완전 보정)를 낮추면 보정 강도가 약해져, 카메라에 가까울 때(bbox 큼)
      과도하게 둔감해지고 멀 때(bbox 작음) 과도하게 예민해지는 양쪽 극단을 평준화할 수 있음.
      (2026-06-16: normalize_gamma=0.6 실험 적용했으나, 화구별 실제 bbox_diag 분포 데이터
      없이 결정한 값이라 근거 부족으로 1.0(완전 보정)으로 환원. 현장에서 `diag_rms.py`로
      bbox_d 분포를 측정해 ref_diag(40px)와 차이가 크게 확인되면 데이터 기반으로 재조정.)
  - RMS EMA 스무딩 (alpha=0.35) + window 투표 (25프레임, 14개 이상 → STEAMING)
  ↓
상태머신 갱신 (core/state_machine.py)
```

**Phase 3 (주파수 분석)**: 시도했으나 실패 → 비활성화. 상세 이유는 ACTION_PLAN.md 참조.

**진단 로그**: 시작 시 카메라별 ROI 합집합 면적(%)이 콘솔에 자동 출력 — 매장 셋업에서 카메라 부담 분배 평가용. 40% 미만 ✓ / 40~65% ⚠ 적정 / 65% 이상 ⚠⚠ 카메라 추가 검토.

---

## 수동 조작 UI

### 화면 레이아웃 (2026-06-16 통합 캔버스 재설계)

좌/우 분리 패널(영상 좌측 + 카드 우측) 구조를 폐지하고, **카메라 박스와 화구 카드가 하나의
캔버스 위에 반칸(half-cell) 단위 그리드로 함께 배치**되는 구조로 변경. 상단 통합 툴바만
고정이고, 그 아래 전체가 캔버스. 화구 카드는 2×2 반칸 고정 크기, 카메라 박스는 칸 수를
자유롭게 지정 가능 — 매장 실제 배치(예: ㄱ자, 두 블록 분리 등)를 그대로 재현할 수 있음.

```
┌──────────────────────────────────────────────────────────┐
│  길가옆에 압력밥솥 타이머                                   │ ← 헤더
├──────────────────────────────────────────────────────────┤
│ ⚙설정(F) 🪄배치(L) Mask(M) 카메라전환(C) Dev(D) 일시정지   │ ← 통합 툴바
│ 영상(V) 소리(N)                        카메라 N대 [-][+]   │
├──────────────────────────────────────────────────────────┤
│  [1] [3] [5] [7] [9]                       [11]   [12]    │
│    [2] [4] [6] [8] [10]                                   │ ← 캔버스
│  ┌──────────┐ ┌──────────┐                 [13]   [14]    │   (반칸 그리드,
│  │  CAM-5   │ │  CAM-6   │                                │    카드+카메라
│  └──────────┘ └──────────┘                       [15]     │    자유 배치)
└──────────────────────────────────────────────────────────┘
```

화구 카드 자체도 재설계됨: **화구 번호가 카드에서 가장 크고 눈에 잘 보이는 핵심 요소**로
상단 대부분을 차지하고, 상태/타이머는 그 아래 보조 정보 한 줄로 표시.

### 배치 모드 (`L` 키) — 카메라/화구 자유 배치

- `L` 키 또는 툴바 "🪄 배치(L)" 버튼으로 진입
- 화구 카드, 카메라 박스 모두 **드래그로 위치 이동** 가능 (좌측상단 기준 스냅 — 드래그 중
  보이는 그림자/위치가 실제 놓일 자리와 정확히 일치)
- 카메라 박스 우하단의 ↘ 핸들을 드래그하면 **반칸 단위로 크기 조절** 가능
- `ENTER` 저장 (화구는 `grid_pos`, 카메라는 `layout_pos`/`layout_size`로 config에 기록) /
  `ESC` 취소
- 화구 카드의 "남은 시간 적은 화구가 위로 뜨는 자동 정렬"은 폐지 — 항상 저장된 위치에 고정 표시

### 키/버튼 동작

| 버튼/키 | 동작 |
|---------|------|
| 화구 카드 클릭 | 해당 화구 선택 (테두리 강조) |
| `R` 버튼 | 선택 화구 타이머 초기화 → EMPTY (길게 누르기 1초) |
| `S` 버튼 | 선택 화구 타이머 강제 진행 (상태별로 동작 다름 — 초벌강제완료/바로재벌/재벌강제시작 등) |
| `ESC` | 선택 해제 / 풀뷰 해제 |
| `1`~`9`, `0` | 1번~10번 화구 선택 |
| `M` | 세그멘테이션 마스크 오버레이 토글 |
| `C` | 카메라 전환 |
| `D` | 개발자 모드 토글 (RMS 수치 등 디버그 정보 표시) |
| `V` | 카메라 박스 표시/숨김 토글 (숨기면 화구 카드만) |
| `N` | 소리(안내음) ON/OFF |
| `Space` | 비디오 일시정지 / 재생 (파일 소스 시) |
| 카메라 영상 더블클릭 | 해당 카메라 풀뷰(단독 확대) 토글 |
| `F` | 인앱 캘리브레이션 모드 진입 (드래그로 ROI 설정) |
| `F` 모드에서 `Enter` | ROI 저장 후 캘리브레이션 종료 |
| `F` 모드에서 `Z` | 마지막 ROI 취소 |
| `F` 모드에서 `[` / `]` | 선택 화구를 같은 카메라 내에서 앞/뒤로 이동 (ID 순서 변경) |
| `F` 모드에서 `Esc` | 캘리브레이션 취소 (저장 안 함) |
| `L` | 배치 모드 진입 (위 "배치 모드" 절 참조) |

> **캘리브레이션**: 별도 `calibration.py` 스크립트 없이 F 키로 메인 UI 내에서 바로 실행.

### 안내음 (사운드)

- 초벌/재벌 완료 30초 전, 완료 시점에 안내음 재생 (`assets/sounds/30초+전.mp3`, `완료.mp3`)
- 화구별 번호 안내음(`assets/sounds/{1~30}번.mp3`)을 완료 안내와 함께 재생해 "몇 번 화구"인지
  소리로도 식별 가능 — 주방처럼 화면을 계속 보고 있기 어려운 환경 대응
- 첫 화구는 지연 없이 즉시 번호를 말하고, 그 음성이 재생되는 동안 같은 상태(30초전/완료)의
  다른 화구가 더 끼어들면 번호만 이어붙여 한 배치로 묶임 (예: "1번 2번 3번 30초전").
  재생이 끝날 때까지 끼어드는 화구가 없으면 그 자리에서 바로 상태어를 붙여 마무리.
  다른 상태가 끼어들면 진행 중인 배치를 즉시 마무리하고 새 배치를 시작 (`_open_status`로 추적)
- `N` 키로 전체 음소거 가능
- `N` 키로 전체 음소거 가능

---

## 파일 구조

```
pressure_timer/
├── CLAUDE.md                    # 시스템 설계 & 불변 지식 (이 파일)
├── ACTION_PLAN.md               # 진행 상황 & 설계 이력 & 미결 과제
├── pyproject.toml
├── requirements.txt
├── main.py                      # 진입점
├── calibration.py               # ROI 캘리브레이션 (드래그 기반)
├── train.py                     # YOLO 학습 스크립트
├── extract_frames.py            # 영상에서 프레임 추출
├── augment_dataset.py           # 데이터 증강 파이프라인
├── diag_rms.py                  # RMS 진단 스크립트 (FP 원인 분석, 화구별 per-frame 출력)
├── core/
│   ├── state_machine.py         # 7상태 머신 (WAIT_SECOND 포함) + 타이머 + 잠금 로직
│   ├── frame_processor.py       # Phase 1+2 통합, body/weight 매칭
│   ├── detector.py              # YOLO-seg 추론 래퍼 (mask_xy 포함)
│   ├── stabilizer.py            # Phase 1: LK+RANSAC+EMA 흔들림 보정
│   ├── optical_flow.py          # Phase 2: Farneback RMS + EMA + window
│   └── frequency_filter.py      # Phase 3: IIR bandpass (현재 비활성화)
├── sources/
│   ├── video_source.py          # 카메라/파일 입력 추상화
│   └── camera_utils.py          # 카메라 전환 유틸
├── ui/
│   └── ui_display.py            # pygame UI (통합 캔버스 — 카메라+화구 카드 반칸 그리드 자유배치, 배치모드, 인앱 캘리브레이션, 안내음 포함)
├── tests/
│   └── compare_phase3.py        # Phase 3 시각 비교 뷰어
├── models/
│   └── pot_seg.pt               # 현재 학습된 모델
├── dataset/
│   └── dataset.yaml
└── config/
    └── store_config.json
```

---

## 환경 설정

### 실행 방법

```bash
# 기본 실행 (카메라)
uv run python main.py

# 내부 영상으로 실행
uv run python main.py --source-0 raw/Side_01.mov

# config 지정
uv run python main.py --config config/store_001.json

# RMS 진단 (정지 노이즈 확인 / threshold 튜닝)
uv run python diag_rms.py --frames 300
uv run python diag_rms.py --burner 3 --frames 150 --skip 30
```

> **캘리브레이션**: 실행 후 F 키 → 영상에서 마우스 드래그로 화구 ROI 지정 → Enter 저장.
> `--calibrate` CLI 플래그는 제거됨 (2026-05-05).

### pyproject.toml

```toml
[project]
name = "pressure-timer"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "ultralytics>=8.0.0",
    "opencv-python>=4.8.0",
    "pygame>=2.5.0",
    "numpy>=1.24.0",
]
```

---

## store_config.json 구조

```json
{
  "store_id": "store_001",
  "sources": [
    {
      "id": 0, "type": "camera", "index": 0,
      "layout_pos": [8, 0], "layout_size": [8, 7]
    }
  ],
  "ui": {
    "layout_cols": 16,
    "window_size": [1280, 720]
  },
  "optical_flow": {
    "rms_threshold": 0.20,
    "rms_ema_alpha": 0.35,
    "window_frames": 25,
    "trigger_frames": 14,
    "normalize_rms": true,
    "normalize_ref_diag": 40.0,
    "normalize_gamma": 1.0
  },
  "frequency": {
    "enabled": false
  },
  "model": {
    "weights": "models/pot_seg.pt",
    "confidence": 0.5
  },
  "burners": [
    {"id": 1, "source_id": 0, "countdown_first": 720, "countdown_second": 270,
     "done_first_timeout": 120, "pot_absent_threshold": 30, "grid_pos": [0, 0]}
  ]
}
```

> **`grid_pos`/`layout_pos`/`layout_size` 단위**: 화면 폭을 `ui.layout_cols`개의 "반칸"으로
> 나눈 격자 좌표 `[row, col]` (정수). 화구 카드는 2×2 반칸 고정. 카메라는 `layout_size`로
> 칸 수(rows, cols)를 자유 지정 — UI의 배치 모드(`L` 키)에서 드래그로 직접 수정 가능하며,
> 저장 시 이 값들로 기록됨.

---

## OS 간 호환성 주의사항

### 1. CLI 인자 덮어쓰기 버그 (2026-03-11)

- **현상**: `apply_source_overrides` 함수 유실 → Windows에서 `--source-0` 인자 무시 + `NameError`
- **원칙**: `.json` 설정 파일은 항상 카메라(`"type": "camera", "index": 0`)를 기본값으로 유지. 테스트용 영상은 CLI 인자(`--source-0`)로 덮어씌워 사용.

### 2. Windows 웹캠 MSMF 에러 (`Error: -1072875772`)

- **현상**: `cv2.VideoCapture(0)` 사용 시 MSMF 백엔드가 프레임을 제대로 가져오지 못함
- **조치**: `sources/video_source.py`에서 `cv2.VideoCapture(index, cv2.CAP_DSHOW)` 로 DirectShow 강제 지정
