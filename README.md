# 압력밥솥 타이머 (Pressure Chicken) ⏱️🍗

**AI 비전 기반 다중 압력밥솥 딸랑이 진동 자동 감지 시스템**

음식점 주방에 설치된 카메라(1~5+대)로 여러 압력밥솥의 딸랑이(추) 움직임을 동시에 모니터링하고, 진동이 감지되면 화구별 카운트다운 타이머를 자동 시작·관리하는 비전 기반 시스템.

**핵심 가치**
- 사람이 일일이 시계 보지 않아도 됨 — 카메라가 보고 알아서 타이머 시작·완료 알림
- 카메라 1대로 여러 화구를 한 번에, 또는 카메라 여러 대로 더 큰 주방 커버
- 인터넷 없이 매장 로컬에서 완결 (학습된 YOLO 모델 + OpenCV)

---

## 🚀 시스템 핵심 아키텍처

전체 파이프라인은 단방향:

```
영상 입력 → Phase 1 → YOLO-seg → Matching → Phase 2 → State Machine → UI
```

| 단계 | 모듈 | 역할 |
|------|------|------|
| ① 입력 | [sources/video_source.py](sources/video_source.py) | 카메라/영상/RTSP를 한 인터페이스로 추상화 (Windows DSHOW 강제) |
| ② **Phase 1: Stabilizer** | [core/stabilizer.py](core/stabilizer.py) | LK + RANSAC + EMA warpAffine — 카메라 떨림 제거 |
| ③ **YOLO-seg** | [core/detector.py](core/detector.py) | ROI 합집합 crop 배치 추론 — `pot_body` / `pot_weight` 감지 |
| ④ Matching | [core/frame_processor.py](core/frame_processor.py) | 화구별 ROI → body → weight 1:1 독점 매칭 + TTL 15프레임 폴백 |
| ⑤ **Phase 2: Optical Flow** | [core/optical_flow.py](core/optical_flow.py) | Farneback dense flow + mask 폴리곤 + residual RMS + EMA + window 투표 |
| ⑥ State Machine | [core/state_machine.py](core/state_machine.py) | 7-state FSM + 타이머 잠금 + pot_absent 디바운스 |
| ⑦ UI | [ui/ui_display.py](ui/ui_display.py) | pygame 통합 대시보드 — 다카메라 그리드 + 인앱 캘리브 + 핫리로드 |

자세한 설계 원칙·상태 정의·파일 구조: [CLAUDE.md](CLAUDE.md)
상세 진행 이력: [ACTION_PLAN.md](ACTION_PLAN.md)
다카메라 UX 작업 로그: [docs/MULTICAM_UX_WORK_LOG.md](docs/MULTICAM_UX_WORK_LOG.md)

---

## 🛠️ 설치 및 실행

### 운영자용 — 매장 노트북 셋업 (1회)

```powershell
# 1. uv 설치 (Python 패키지 매니저)
winget install astral-sh.uv

# 2. 저장소 받기
git clone <repo-url>
cd Pressure_Chicken

# 3. 의존성 자동 설치 (수 분 소요)
uv sync
```

### 매일 실행

**바탕화면에 `run.bat` 바로가기를 만들어 더블클릭하면 끝.**

```powershell
# 또는 명령줄로:
.\run.bat
```

`run.bat`은 `uv` 설치 확인 → `.venv` 없으면 자동 sync → `main.py` 실행 순서로 진행됨.

### 개발/테스트 시나리오

```powershell
# 기본 실행 (config/store_config.json 자동 로드)
uv run python main.py

# 다른 config 지정
uv run python main.py --config config/examples/store_4cam.json

# 영상 파일로 시뮬레이션 (실제 카메라 없이)
uv run python main.py --source-0 raw/Side_01.mov

# 다중 영상 시뮬 — 4카메라 환경 재현
uv run python main.py --config config/examples/store_4cam_video.json

# 헤드리스 + 스크린샷 (UI 검증용)
uv run python main.py --config <cfg> --test 30 --headless --screenshot out.png
```

---

## 🎮 운영 단축키 & UI

### 화면 구성

```
┌────────────────────────────────┬──────────────────────┐
│   카메라 그리드 (좌측, N대 균등 분할)  │  우측 패널            │
│   ┌──────┬──────┐               │  ─ 설정 / 카메라 +/-  │
│   │Cam-0 │Cam-1 │               │  ─ 화구 카드 그리드     │
│   ├──────┼──────┤               │   #1[Cam-0] 03:42    │
│   │Cam-2 │Cam-3 │               │   #2[Cam-0] 대기      │
│   └──────┴──────┘               │   ...                │
└────────────────────────────────┴──────────────────────┘
```

### 단축키

| 키 | 동작 |
|----|------|
| `F` | 캘리브레이션 모드 진입 (각 카메라 영역에 드래그로 화구 ROI 지정) |
| 캘리브 중 `Enter` | 저장 후 종료 (핫 리로드) |
| 캘리브 중 `Z` | 마지막 화구 취소 |
| 캘리브 중 `Del` / `Backspace` | 선택된 화구 삭제 |
| 캘리브 중 `[` / `]` | 선택된 화구를 같은 카메라 안에서 ID ↓ / ↑ 한 칸 이동 |
| 캘리브 중 `Esc` | 저장 없이 취소 |
| `1`~`9`, `0` | 화구 1~10번 직접 선택 |
| `R` | 선택 화구 리셋 (1초 길게 — 실수 방지) |
| `S` | 선택 화구 타이머 강제 시작 |
| `M` | mask 오버레이 토글 |
| `D` | 개발 모드 (RMS 수치 등) 토글 |
| `Space` | 비디오 일시정지 (파일 소스 시) |
| `C` | 카메라 전환 |
| `Q` | 종료 |

### 화구 ID 부여 정책 (자동 그룹화)

저장(`Enter`) 누르는 순간 **카메라 순서대로 그룹화**되어 ID 1부터 재부여.

```
그릴 때:                          저장 후:
  Cam-0[A,B]                        Cam-0 → ID 1, 2
  Cam-2[C]                          Cam-1 → ID 3
  Cam-1[D]      ←─ 자동 그룹화 ─→   Cam-2 → ID 4
  Cam-3[E]                          Cam-3 → ID 5
```

같은 카메라 안 순서는 **그린 순서** 유지. 바꾸고 싶으면 캘리브 모드에서 화구 클릭 → `[` / `]` 키로 한 칸씩 이동.
**운영 중에는 ID 고정** (다음 캘리브 모드 진입까지).

### 우측 패널 버튼

- **`⚙ 설정(F)`** : 캘리브레이션 모드 진입
- **`Mask(M)`** : segmentation 마스크 오버레이 토글
- **`Dev(D)`** : 개발 모드 (RMS·점수 등 표시)
- **`+ 추가`** : 빈 카메라 인덱스를 찾아 자동 추가 (이미 열린 카메라는 안 건드림)
- **`− 제거`** : **영상 영역에서 카메라를 먼저 클릭해 선택** → 노란 외곽 → 버튼이 `- Cam-N` 빨강으로 바뀜 → 누르면 그 카메라 제거 (묶인 화구도 함께)

### 카메라 선택 방식

| 동작 | 결과 |
|------|------|
| 카메라 셀 영상 영역 클릭 | 그 카메라 선택 → 외곽 노란 4px 스트로크 |
| 같은 카메라 다시 클릭 | 선택 해제 |
| 다른 카메라 클릭 | 선택 이동 |
| `ESC` 키 | 선택 해제 |
| `F` 키 (캘리브 진입) | 자동 선택 해제 |

---

## ⚙️ 매장별 설정 (config/store_config.json)

### 핵심 구조

```json
{
  "sources": [
    {"id": 0, "type": "camera", "index": 0, "label": "1구역", "exposure": -5},
    {"id": 1, "type": "camera", "index": 1, "label": "2구역", "exposure": -5}
  ],
  "burners": [
    {"id": 1, "source_id": 0, "roi": [200, 200, 400, 400],
     "countdown_first": 720, "countdown_second": 300}
  ],
  "optical_flow": {
    "rms_threshold": 0.20,
    "window_frames": 17,
    "trigger_frames": 9
  }
}
```

### 가장 자주 만지는 값

| 키 | 의미 | 권장 |
|----|------|------|
| `sources[].index` | OS가 인식한 카메라 번호 | 0번부터 |
| `sources[].exposure` | 카메라 노출 (자동 노출 OFF) | `-3` ~ `-7` 매장 광량에 맞춰 |
| `burners[].roi` | `[x, y, w, h]` 화구 영역 | **F키로 자동 지정 권장** |
| `burners[].source_id` | 화구가 속한 카메라 ID | 캘리브 시 자동 |
| `optical_flow.rms_threshold` | 진동 인정 임계값 | 0.20 (낮을수록 민감) |

### 3계층 합성 우선순위
`global < source.optical_flow < burner.optical_flow`
→ 카메라마다 / 화구마다 다른 임계값을 일부만 override 가능.

### 예시 config

| 파일 | 용도 |
|------|------|
| [config/examples/store_1cam.json](config/examples/store_1cam.json) | 카메라 1대 매장 |
| [config/examples/store_3cam.json](config/examples/store_3cam.json) | 카메라 3대 매장 |
| [config/examples/store_4cam.json](config/examples/store_4cam.json) | 카메라 4대 매장 (현재 기준) |
| [config/examples/store_5cam.json](config/examples/store_5cam.json) | 카메라 5대 매장 |
| [config/examples/store_4cam_video.json](config/examples/store_4cam_video.json) | 4영상 시뮬 (테스트용) |

---

## 📊 처리 비용 진단

실행 시 콘솔에 자동 출력되는 진단 로그:

```
[main] target_fps=10.0 (cams=4)
[diag] Cam-0: burners=2, ROI 합집합=34.7%  → ✓ 가벼움
[diag] Cam-1: burners=2, ROI 합집합=34.7%  → ✓ 가벼움
[diag] Cam-2: burners=2, ROI 합집합=78.3%  → ⚠⚠ 카메라 추가 검토
[diag] Cam-3: 화구 미할당
```

### 해석

- **ROI 합집합** = 각 카메라에서 화구 ROI들의 픽셀 합 ÷ 화면 픽셀 수 (0~100%)
- YOLO 추론 비용은 이 면적에 거의 비례 → **단일 비용 절감의 최대 레버**
- 🟢 **40% 미만** : 가벼움 (안정 운영)
- 🟡 **40~65%** : 적정
- 🔴 **65% 이상** : 카메라 추가 분담 검토 (한 카메라가 화면 대부분을 화구로 가득 채우면 부담 ↑)

### 비용 결정 4축

1. **카메라 수** → frame I/O + Stabilizer + YOLO 호출 횟수
2. **ROI 합집합 면적** → YOLO 추론 비용 (가장 큰 레버)
3. **화구 수** → Phase 2 호출 횟수
4. **딸랑이 bbox 크기** → Phase 2 호출당 Farneback 면적

### 자동 최적화 (코드 내장)

- **target_fps 자동**: 1~2대=15fps, 3~6대=10fps, 7대+=8fps
- **ROI 합집합 crop 배치 추론**: 카메라당 YOLO 1회 호출
- **동적 margin**: ROI 짧은 변의 10% (안전선 30~50px)
- **TTL 15프레임 폴백**: 일시 가림에 강건

---

## 🎯 매장 셋업 절차 (권장)

1. **카메라 4대 USB 연결**
   - 노트북 좌/우/후면 / USB-C에 **분산**해서 꽂기 (대역폭 경쟁 방지)
   - 가능하면 외장 전원 USB 3.0 허브
   - 케이블은 빠짐 방지로 고정

2. **노트북 절전 모드 OFF**
   - 전원 옵션 → "절대 절전 안 함"

3. **`run.bat` 실행** → 4 카메라 그리드 표시

4. **F키로 캘리브레이션**
   - 각 카메라 영역에 화구 위치를 마우스 드래그
   - 잘못 그렸으면 클릭 후 `Del`
   - `Enter` 저장 → 핫 리로드

5. **콘솔 진단 로그 확인**
   - 모든 카메라가 🟢 또는 🟡이면 OK
   - 🔴이면 카메라 추가 또는 화구 분담 재배치

6. **카메라 노출 수동 고정** (`config/store_config.json`의 `sources[].exposure`)
   - 자동 노출은 false positive의 주범 — 매장 광량에 맞춰 `-3` ~ `-7` 시도

7. **20~30초 워밍업 후 운영 시작**
   - 카메라 첫 프레임은 AWB·노출이 출렁임

---

## 🎓 학습 (필요 시)

새 데이터 추가하여 모델 재학습:

```powershell
# 영상에서 프레임 추출
uv run python extract_frames.py

# 데이터 증강
uv run python augment_dataset.py

# 학습 (yolov8n-seg 베이스, 최초 1회 자동 다운로드)
uv run python train.py
```

현재 모델 성능 (2026-04-12 기준):
- mAP50(B) **85.0%** / mAP50(M) **81.0%**
- 파일: `models/pot_seg.pt` (6.5MB, YOLOv8n-seg, 3.26M params)

---

## 📁 프로젝트 구조

```
Pressure_Chicken/
├── README.md                    # 이 파일
├── CLAUDE.md                    # 시스템 설계 & 불변 지식 (가장 신뢰 가능)
├── ACTION_PLAN.md               # 진행 이력 & 미결 과제
├── MULTI_CAMERA_PLAN.md         # 다카메라 확장 설계
├── run.bat                      # 운영자용 더블클릭 실행 스크립트
├── main.py                      # 진입점
├── train.py                     # 학습 스크립트
├── core/                        # 감지 파이프라인 (Phase 1+2 + FSM)
├── sources/                     # 영상 입력 (카메라/파일)
├── ui/                          # pygame 통합 대시보드
├── config/
│   ├── store_config.json        # 운영 설정 (매장별)
│   └── examples/                # 카메라 수별 예시 config
├── models/
│   └── pot_seg.pt               # 학습된 segmentation 모델
└── docs/                        # 파이프라인 다이어그램, 작업 로그
```

---

## 🆘 자주 발생하는 문제

| 증상 | 원인/대처 |
|------|-----------|
| `Error: -1072875772` (Windows) | MSMF 백엔드 오류 — 이미 DSHOW 강제로 자동 처리됨 |
| 카메라가 안 잡힘 | 다른 앱(Teams, Zoom)이 점유 중인지 확인 |
| 화면이 검고 ⛔ 신호 없음 | USB 다시 꽂기 → 5초 대기 → 안 되면 프로그램 재시작 |
| 정지 상태인데 진동 감지됨 | 자동 노출이 원인 가능 — `exposure` 값 수동 고정 |
| ROI 합집합 65% 초과 경고 | 화구를 다른 카메라로 분담하거나 카메라 추가 |
| AI 모델 없음 경고 | `models/pot_seg.pt` 가 git에 있는지 확인 |

---

## 📜 라이센스 & 기여

내부 프로젝트.

**관련 문서**
- 시스템 설계: [CLAUDE.md](CLAUDE.md)
- 진행 이력: [ACTION_PLAN.md](ACTION_PLAN.md)
- 다카메라 설계: [MULTI_CAMERA_PLAN.md](MULTI_CAMERA_PLAN.md)
- 다카메라 UX 작업 로그: [docs/MULTICAM_UX_WORK_LOG.md](docs/MULTICAM_UX_WORK_LOG.md)
- 파이프라인 다이어그램: [docs/pipeline.png](docs/pipeline.png) · [docs/phase1.png](docs/phase1.png) · [docs/phase2.png](docs/phase2.png)
