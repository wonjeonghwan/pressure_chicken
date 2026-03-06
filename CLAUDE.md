# 압력밥솥 타이머 시스템 (Pressure Cooker Timer)

## 프로젝트 개요

음식점 주방에서 여러 개의 압력밥솥을 동시에 운영할 때,
각 화구의 압력밥솥 딸랑이(추)가 움직이기 시작하면 자동으로 카운트다운 타이머를 시작하는
비전 기반 모니터링 시스템.

**인터넷 연결**: 학습 시 `yolov8n.pt` 베이스 모델 최초 1회 다운로드만 필요.
이후 실행(운영)은 인터넷 없이 완전히 로컬에서 동작.

---

## 감지해야 할 것 (핵심)

1. **화구에 밥솥이 있는가 없는가**
2. **딸랑이가 밥솥 몸체 기준으로 상대적으로 움직이고 있는가**
   - 밥솥 전체가 이동한 것과 딸랑이만 진동하는 것을 구분
   - 연기/가림으로 인한 픽셀 변화와 실제 딸랑이 진동을 구분
3. **타이머가 한 번 시작되면 외부 감지 결과와 무관하게 독립 동작**

---

## 화구 상태 정의

| 상태 | 설명 | UI 색상 |
|------|------|---------|
| `EMPTY` | 빈 화구 (밥솥 미감지) | 회색 |
| `POT_IDLE` | 밥솥 감지, 딸랑이 정지 | 파란색 |
| `POT_STEAMING` | 딸랑이 진동 확정 → 타이머 진행 중 | 초록색 |
| `DONE` | 타이머 완료 | 빨간색 점멸 |

---

## YOLO 학습 클래스 (3가지)

```yaml
# dataset/dataset.yaml
path: ./dataset
train: images/train
val: images/val

nc: 3
names:
  0: empty_burner   # 빈 화구 (밥솥 없음)
  1: pot_body       # 밥솥 몸체 (기준점 역할)
  2: pot_weight     # 딸랑이 (추)
```

**왜 3개인가:**
- `empty_burner`: 빈 화구를 명시적으로 학습 → 오탐 감소
- `pot_body`: 밥솥 전체 이동의 기준점 → 딸랑이의 상대적 움직임 계산에 필수
- `pot_weight`: 딸랑이 위치 추적 → 진동 패턴 감지

---

## 감지 구조

### 딸랑이 진동 판별 로직

단순 픽셀 diff 대신 **딸랑이 중심점의 상대적 이동**을 추적.

```
매 프레임:
  pot_body 중심점 (bx, by) 추출
  pot_weight 중심점 (wx, wy) 추출
          ↓
  상대 위치 계산:
    rel_x = wx - bx
    rel_y = wy - by
          ↓
  rel_x, rel_y 의 시계열 변화 추적
          ↓
  판별:
    pot_body 고정 + rel_x 좌우 반복 진동 → 딸랑이 진동 (타이머 트리거)
    pot_body 크게 이동                   → 밥솥 전체 이동 → 무시
    pot_weight confidence 급락/소멸      → 연기/가림 → 무시
```

### 진동 확정 조건 (오탐 방지)

```
60프레임 윈도우 내에서:
  - pot_body 이동량 < 10px (밥솥 고정 확인)
  - pot_weight confidence >= 0.5
  - rel_x 변화가 ±3px 이상 진동을 40프레임 이상 감지
          ↓
  → 진동 확정 → POT_STEAMING 진입
```

단순 "N프레임 연속" 방식 대신 **윈도우 내 비율** 방식 사용.
1프레임 끊겨도 리셋되지 않음.

### 전체 처리 흐름

```
영상 프레임 입력
      ↓
YOLO 추론
  ├─ empty_burner 감지 or 아무것도 없음 → 상태: EMPTY
  ├─ pot_body만 감지 (pot_weight 없음)  → 상태: POT_IDLE (밥솥은 있으나 딸랑이 미확인)
  └─ pot_body + pot_weight 모두 감지
          ↓
    상대 위치(rel_x, rel_y) 계산
          ↓
    진동 확정 조건 충족?
    ├─ No  → 상태: POT_IDLE
    └─ Yes → 상태: POT_STEAMING, 타이머 시작
                    ↓
              [타이머 잠금 — 이후 감지 결과 무시]
                    ↓
              타이머 0초 도달 → DONE (점멸)
                    ↓
              수동 초기화 or 자동 초기화 → EMPTY
```

### 타이머 잠금 규칙

```
POT_STEAMING 진입 순간:
  - 해당 화구의 감지 루프 결과를 상태 변경에 사용하지 않음
  - 타이머만 독립적으로 tick
  - 사람이 가리든, 연기가 생기든, 밥솥을 잠깐 들든 → 타이머 유지
  - 오직 수동 초기화(R키 or UI 버튼)만 타이머 중단 가능
```

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
├──────────┼──────────┼──────────┤                 │
│   4번    │   5번    │   6번    │                 │
│  대기    │  1:15    │  대기    │                 │
│  ⟳  ▶  │  ⟳  ▶  │  ⟳  ▶  │                 │
└──────────┴──────────┴──────────┴─────────────────┘
[선택: 1번 화구]  R: 초기화   S: 수동시작   ESC: 선택해제
```

### 버튼 동작

| 버튼/키 | 동작 |
|---------|------|
| 화구 카드 클릭 | 해당 화구 선택 (테두리 강조) |
| `⟳` 버튼 or `R` | 선택 화구 타이머 초기화 → EMPTY |
| `▶` 버튼 or `S` | 선택 화구 타이머 강제 시작 → POT_STEAMING |
| `ESC` | 선택 해제 |
| `1`~`9`, `0` | 1번~10번 화구 선택 (11번 이상은 마우스 클릭으로 선택)

### UI 주의사항
- `⟳`(초기화) 버튼은 실수 방지를 위해 **클릭 후 1초 내 재확인** 또는 **길게 누르기(1초)** 방식으로 구현
- `▶`(수동시작)은 단순 클릭으로 즉시 동작
- DONE 상태(빨간 점멸) 화구는 클릭 한 번으로 바로 초기화 가능 (확인 불필요)

---

## 파일 구조

```
pressure_timer/
├── CLAUDE.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .python-version
├── main.py                      # 진입점
├── train.py                     # YOLO 학습 스크립트
├── extract_frames.py            # 영상에서 프레임 추출
├── core/
│   ├── __init__.py
│   ├── state_machine.py         # 화구별 상태 + 타이머 (잠금 로직 포함)
│   ├── frame_processor.py       # YOLO + 상대좌표 계산 + 진동 판별
│   └── detector.py              # YOLO 추론 래퍼
├── sources/
│   ├── __init__.py
│   └── video_source.py          # 카메라/파일 입력 추상화
├── ui/
│   ├── __init__.py
│   └── ui_display.py            # pygame UI (수동 조작 포함)
├── models/
│   └── .gitkeep
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── dataset.yaml
├── runs/
└── config/
    └── store_config.json
```

---

## 환경 설정

### uv 사용 (권장)

```bash
# uv 설치 (최초 1회)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 셋업
cd pressure_timer
uv sync

# 실행
uv run python main.py
uv run python train.py
```

### pip 사용 (대안)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
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

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### requirements.txt

```
ultralytics>=8.0.0
opencv-python>=4.8.0
pygame>=2.5.0
numpy>=1.24.0
```

### .python-version

```
3.11
```

### .gitignore

```
.venv/
runs/
models/*.pt
dataset/images/
dataset/labels/
*.mp4
*.avi
*.mov
__pycache__/
*.pyc
.DS_Store
config/store_*.json
```

---

## Step 1. 영상에서 프레임 추출

```bash
uv run python extract_frames.py --video 녹화영상.mp4 --fps 2
```

`extract_frames.py`:
```python
import cv2, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True)
parser.add_argument('--fps', type=float, default=2)
args = parser.parse_args()

os.makedirs('dataset/images/train', exist_ok=True)

cap = cv2.VideoCapture(args.video)
video_fps = cap.get(cv2.CAP_PROP_FPS)
interval = max(1, int(video_fps / args.fps))  # 최소 1 보장

count, saved = 0, 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count % interval == 0:
        cv2.imwrite(f'dataset/images/train/frame_{saved:04d}.jpg', frame)
        saved += 1
    count += 1

cap.release()
print(f'{saved}장 추출 완료 → dataset/images/train/')
```

---

## Step 2. 라벨링

1. [roboflow.com](https://roboflow.com) → 무료 계정 생성
2. New Project → Object Detection
3. `dataset/images/train/` 이미지 전체 업로드
4. 클래스 3개 등록 (순서 중요): `empty_burner`, `pot_body`, `pot_weight`
5. 박스 라벨링:
   - 빈 화구 전체 영역 → `empty_burner`
   - 밥솥 몸체(뚜껑 포함) 영역 → `pot_body`
   - 딸랑이(추) 영역만 → `pot_weight`
6. Generate → train 80% / val 20% 자동 분리
7. Export → **YOLOv8 형식** 다운로드
8. 다운받은 내용을 `dataset/` 폴더에 덮어쓰기
9. `dataset/dataset.yaml`의 클래스 순서가 `empty_burner(0)`, `pot_body(1)`, `pot_weight(2)` 인지 반드시 확인

---

## Step 3. 학습

```bash
uv run python train.py
```

`train.py`:
```python
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch

model = YOLO('yolov8n.pt')  # 최초 실행 시 인터넷 필요 (자동 다운로드)

results = model.train(
    data='dataset/dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=8,        # 메모리 부족 시 4로 줄이기
    device='0' if torch.cuda.is_available() else 'cpu',
    project='runs',
    name='pot_detector',
    exist_ok=True,
)

best = Path('runs/pot_detector/weights/best.pt')
if best.exists():
    Path('models').mkdir(exist_ok=True)
    shutil.copy(best, 'models/pot_detector.pt')
    print("✅ 학습 완료. models/pot_detector.pt 저장됨")
else:
    print("❌ 학습 실패. runs/pot_detector/weights/ 확인 필요")
```

**학습 시간 참고:**
- 노트북 CPU: 약 1~3시간 (데이터 200장 기준)
- 노트북 GPU (CUDA): 약 10~20분
- 전원 연결 필수

---

## Step 4. 실행

```bash
# 녹화 영상으로 테스트
uv run python main.py --source-0 video_a.mp4

# 카메라 실시간
uv run python main.py

# 매장 config 지정
uv run python main.py --config config/store_001.json
```

---

## store_config.json 구조

```json
{
  "store_id": "store_001",
  "store_name": "매장명",
  "sources": [
    {"id": 0, "type": "camera", "index": 0},
    {"id": 1, "type": "camera", "index": 1}
  ],
  "ui": {
    "grid_cols": 6,
    "window_size": [1280, 720]
  },
  "motion": {
    "window_frames": 60,
    "trigger_frames": 40,
    "rel_x_threshold": 5,
    "body_move_limit": 10,
    "min_confidence": 0.5
  },
  "model": {
    "weights": "models/pot_detector.pt",
    "confidence": 0.5
  },
  "burners": [
    {"id": 1, "source_id": 0, "countdown_seconds": 300, "grid_pos": [0, 0]},
    {"id": 2, "source_id": 0, "countdown_seconds": 300, "grid_pos": [0, 1]}
  ]
}
```

---

## 구현 순서 (클로드코드 작업 순서)

1. `core/state_machine.py` — 상태 정의, 타이머, 잠금 로직, 수동 조작 메서드
2. `sources/video_source.py` — 카메라/파일 입력 추상화
3. `core/detector.py` — YOLO 추론 래퍼 (클래스별 bounding box 반환)
4. `core/frame_processor.py` — 상대좌표 계산, 진동 윈도우 판별
5. `ui/ui_display.py` — pygame UI, 화구 카드, 수동 조작 버튼/키
6. `extract_frames.py` — 프레임 추출 유틸
7. `train.py` — 학습 스크립트
8. `main.py` — 전체 통합

---

## 새 매장 추가 시

1. 그 매장에서 영상 촬영
2. `extract_frames.py`로 프레임 추출
3. 기존 `dataset/images/train/`에 추가
4. Roboflow에서 추가 라벨링 후 재export
5. `uv run python train.py` 재학습
6. `models/pot_detector.pt` 교체
7. 해당 매장 `store_config.json` 작성 후 실행
