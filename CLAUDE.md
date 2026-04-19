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
초벌 완료 → 딸랑이 재감지 → 재벌 5분 시작
재벌 완료 → 경보
밥솥 이탈 (어느 단계든) → 전체 리셋 → 새 사이클
```

### 타이머 잠금
- 타이머 진행 중에는 카메라 감지 결과로 상태가 자동 변경되지 않음
- 딸랑이가 가려졌다가 다시 보여도 "또 시작"으로 오인하지 않음
- **수동 조작만** 상태를 바꿀 수 있음 (R키 초기화, S키/▶버튼 강제시작)

---

## 화구 상태 정의 (6개)

| 상태 | 설명 | UI 색상 |
|------|------|---------|
| `EMPTY` | 빈 화구 (밥솥 미감지) | 회색 |
| `POT_IDLE` | 밥솥 감지, 대기 중 | 파란색 |
| `POT_STEAMING_FIRST` | 초벌 12분 타이머 진행 중 | 초록색 |
| `DONE_FIRST` | 초벌 완료, 재벌 딸랑이 대기 | 노란색 |
| `POT_STEAMING_SECOND` | 재벌 5분 타이머 진행 중 | 진한 초록 |
| `DONE_SECOND` | 재벌 완료, 경보 | 빨간색 점멸 |

---

## YOLO 학습 클래스 (3가지)

```yaml
# dataset/dataset.yaml
nc: 3
names:
  0: empty_burner   # 빈 화구 (밥솥 없음)
  1: pot_body       # 밥솥 몸체 (기준점 역할)
  2: pot_weight     # 딸랑이 (추)
```

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
YOLO-seg 배치 추론 (ROI 전체 1회 호출, 15fps)
  ↓
body/weight 매칭 (ROI 중심거리 기반 그리디)
  ↓
Phase 2: OpticalFlowDetector (core/optical_flow.py)
  - Farneback dense flow
  - crop 위치: bbox center EMA (pos_alpha=0.3) — mask 유무 무관하게 항상 bbox center 기준
  - RMS 계산: mask_xy 폴리곤 내부 픽셀만 (mask 없는 프레임은 bbox 전체 fallback)
  - RMS EMA 스무딩 (alpha=0.35) + window 투표 (25프레임, 14개 이상 → STEAMING)
  ↓
상태머신 갱신 (core/state_machine.py)
```

**Phase 3 (주파수 분석)**: 시도했으나 실패 → 비활성화. 상세 이유는 ACTION_PLAN.md 참조.

---

## 수동 조작 UI

### 화면 레이아웃

```
┌─────────────────────────────────────────────────┐
│           압력밥솥 타이머 모니터                  │
├──────────┬──────────┬──────────┬──────────────── │
│   1번    │   2번    │   3번    │   ...           │
│  3:42    │  대기    │  완료!   │                 │
│  ⟳  ▶  │  ⟳  ▶  │  ⟳  ▶  │                 │
└──────────┴──────────┴──────────┴─────────────────┘
[선택: 1번 화구]  R: 초기화   S: 수동시작   ESC: 선택해제
```

### 키/버튼 동작

| 버튼/키 | 동작 |
|---------|------|
| 화구 카드 클릭 | 해당 화구 선택 (테두리 강조) |
| `⟳` 버튼 or `R` | 선택 화구 타이머 초기화 → EMPTY (길게 누르기 1초) |
| `▶` 버튼 or `S` | 선택 화구 타이머 강제 시작 → POT_STEAMING (즉시) |
| `ESC` | 선택 해제 |
| `1`~`9`, `0` | 1번~10번 화구 선택 |
| `M` | 세그멘테이션 마스크 오버레이 토글 |
| `C` | 카메라 전환 |

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
│   ├── state_machine.py         # 6상태 머신 + 타이머 + 잠금 로직
│   ├── frame_processor.py       # Phase 1+2 통합, body/weight 매칭
│   ├── detector.py              # YOLO-seg 추론 래퍼 (mask_xy 포함)
│   ├── stabilizer.py            # Phase 1: LK+RANSAC+EMA 흔들림 보정
│   ├── optical_flow.py          # Phase 2: Farneback RMS + EMA + window
│   └── frequency_filter.py      # Phase 3: IIR bandpass (현재 비활성화)
├── sources/
│   ├── video_source.py          # 카메라/파일 입력 추상화
│   └── camera_utils.py          # 카메라 전환 유틸
├── ui/
│   └── ui_display.py            # pygame UI (수동 조작 포함)
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

# 내부 영상으로 캘리브레이션
uv run python main.py --source-0 raw/Side_01.mov --calibrate

# 내부 영상으로 실행
uv run python main.py --source-0 raw/Side_01.mov

# config 지정
uv run python main.py --config config/store_001.json

# N프레임 후 자동 종료 (테스트용)
uv run python main.py --test 60
```

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
    {"id": 0, "type": "camera", "index": 0}
  ],
  "ui": {
    "grid_cols": 6,
    "window_size": [1280, 720]
  },
  "optical_flow": {
    "rms_threshold": 0.5,
    "rms_ema_alpha": 0.35,
    "window_frames": 25,
    "trigger_frames": 14
  },
  "frequency": {
    "enabled": false
  },
  "model": {
    "weights": "models/pot_seg.pt",
    "confidence": 0.5
  },
  "burners": [
    {"id": 1, "source_id": 0, "countdown_first": 720, "countdown_second": 300, "grid_pos": [0, 0]}
  ]
}
```

---

## OS 간 호환성 주의사항

### 1. CLI 인자 덮어쓰기 버그 (2026-03-11)

- **현상**: `apply_source_overrides` 함수 유실 → Windows에서 `--source-0` 인자 무시 + `NameError`
- **원칙**: `.json` 설정 파일은 항상 카메라(`"type": "camera", "index": 0`)를 기본값으로 유지. 테스트용 영상은 CLI 인자(`--source-0`)로 덮어씌워 사용.

### 2. Windows 웹캠 MSMF 에러 (`Error: -1072875772`)

- **현상**: `cv2.VideoCapture(0)` 사용 시 MSMF 백엔드가 프레임을 제대로 가져오지 못함
- **조치**: `sources/video_source.py`에서 `cv2.VideoCapture(index, cv2.CAP_DSHOW)` 로 DirectShow 강제 지정
