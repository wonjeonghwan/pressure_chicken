# 압력밥솥 타이머 — 액션플랜

> 최종 업데이트: 2026-03-10 (3차) | 현재 단계: Phase 2 진동 감지 로직 튜닝 및 검증 진행 중

---

## 확정된 설계 원칙

### 타이머 사이클
```
밥솥 올라옴 → 딸랑이 2~3초 이상 감지 → 초벌 12분 시작
초벌 12분 완료 → 딸랑이 재감지 → 재벌 5분 시작
재벌 5분 완료 → 경보
밥솥 이탈(어느 단계든) → 전체 리셋 → 새 사이클
```

### 타이머 잠금
- 타이머 진행 중에는 카메라 감지 결과로 상태가 자동 변경되지 않음
- 딸랑이가 가려졌다가 다시 보여도 "또 시작"으로 오인하지 않음
- **수동 조작만** 상태를 바꿀 수 있음

### 카메라 구성
- 테스트: 카메라 1대, 화구 10개 이내
- 운영: 카메라 N대, 카메라당 약 10개 화구
- config만 수정하면 확장 가능 (코드 변경 없음)

---

## Phase 0 — 데이터 준비 및 라벨링

### Step 0-1. 영상에서 프레임 추출

```bash
uv run python extract_frames.py --video 주방영상.mp4 --fps 2
```

목표: **100장 이상** (Roboflow 자동 라벨링 품질 확보)

### Step 0-2. Roboflow 자동 라벨링 (50장)

1. [roboflow.com](https://roboflow.com) 접속
2. New Project → Object Detection
3. `dataset/images/train/` 에서 **50장** 업로드
4. **Auto Label** 기능 사용 (Grounding DINO 기반)
5. 자동 결과 검토 → 틀린 것 수정
6. 클래스 3개 확인:
   - `0: empty_burner` — 빈 화구
   - `1: pot_body` — 밥솥 몸체
   - `2: pot_weight` — 딸랑이(추)
7. YOLOv8 형식으로 Export → `dataset/` 폴더에 덮어쓰기

### Step 0-3. 1차 학습 (부트스트랩)

```bash
uv run python train.py
```

목적: 나머지 자동 라벨링용 모델 확보 (정확도 낮아도 됨)

### Step 0-4. 나머지 프레임 자동 라벨링

```bash
uv run python auto_label.py --input dataset/images/train/ --model models/pot_detector.pt
```

> `auto_label.py` — Phase 1 에서 구현 예정

자동 라벨링 결과를 Roboflow에 업로드 → 검토 후 최종 라벨 확정

### Step 0-5. 최종 학습

```bash
uv run python train.py
```

목표 지표:
- `mAP@0.5 > 0.85` (pot_body, pot_weight 클래스)
- 연기/가림 환경 샘플 포함 확인

---

## Phase 1 — 코드 업데이트

### Step 1-1. 상태머신 확장

`core/state_machine.py` — 상태 6개로 확장 완료

| 상태 | 설명 | UI 색상 |
|------|------|---------|
| `EMPTY` | 밥솥 없음 | 회색 |
| `POT_IDLE` | 밥솥 있음, 대기 | 파란색 |
| `POT_STEAMING_FIRST` | 초벌 12분 진행 중 | 초록색 |
| `DONE_FIRST` | 초벌 완료, 재벌 대기 | 노란색 |
| `POT_STEAMING_SECOND` | 재벌 5분 진행 중 | 진한 초록 |
| `DONE_SECOND` | 재벌 완료, 경보 | 빨간색 점멸 |

### Step 1-8. 진동 감지 로직 고도화 (2026-03-09)
- 하이브리드 방식 도입: YOLO(위치고정) + Pixel Diff(상세모션)
- EMA(지수이동평균) 적용으로 YOLO 박스 Jitter 제거
- 템플릿 매칭(`matchTemplate`)으로 증기/가림 현상과 실제 진동 구분
- `test_motion.py`를 통한 특정 구간(Stationary vs Moving) 검증 환경 구축

### Step 1-9. `auto_label.py` 구현 (예정)
YOLO 모델로 이미지를 추론하여 YOLO 형식 라벨 파일 자동 생성.

### Step 1-10. config 유효성 검사 (예정)
실행 시 config 오류를 사람이 읽기 쉬운 메시지로 출력.

---

## Phase 2 — 테스트 (카메라 1대 / 녹화 영상)

**현재 상태**: 시스템 실행 및 멀티 화구 감지 가능. 하이브리드 모드 적용 후 오탐율 대폭 감소.
**진행 중**: 실제 현장 영상(`DJI_20260304141645_0001_D.MP4`)의 STATIONARY/MOVING 구간 정밀 튜닝 중.

### 체크리스트
- [x] ROI 재캘리브레이션 (전체 화구 완료)
- [/] 감지 정확도 튜닝 (HybridVibrationTracker 파라미터 최적화)
- [ ] 딸랑이 진동 → STEAMING 자동 전환 확인
- [ ] 사이클 전환 체크리스트 (초벌 -> 대기 -> 재벌)
- [ ] 타이머 잠금 체크리스트 (사람 가림, 연기 발생 시 유지)
- [ ] 수동 조작 체크리스트 (시작/초기화 버튼)

---

## Phase 3 — 확장 (카메라 2대)

- [ ] 2대 동작 확인 (config 설정 완료)
- [ ] 20개 화구 동시 실행 성능 (30fps 유지 목표)

---

## 주요 설계 변경 이력

| 날짜 | 파일 | 변경 내용 |
|------|------|----------|
| 2026-03-06 | core/state_machine.py | 4상태 → 6상태 (STEAMING_FIRST/SECOND, DONE_FIRST/SECOND) |
| 2026-03-06 | calibration.py | 화구 수 고정 → 드래그 기반 동적 지정 방식으로 재작성 |
| 2026-03-07 | core/frame_processor.py | read_frames() / detect_and_update() 분리; 배치 YOLO 추론 |
| 2026-03-07 | main.py | 15fps 영상읽기 / 3fps YOLO 분리; FPS 카운터 추가 |
| 2026-03-09 | core/frame_processor.py | `HybridVibrationTracker` 도입: EMA 보정, 템플릿 매칭 기반 진동 판별 |
| 2026-03-09 | test_motion.py | 구간별(Stationary/Moving) 정밀 분석용 스크립트 추가 |
| 2026-03-10 | ACTION_PLAN.md | 현재 상태 업데이트 및 Phase 1-8 추가 |

---

## 현재 진행 위치 상세 (Phase 0/1 완료 내역)

- [x] Phase 0-4b : 재학습 완료 — mAP@0.5=95.3% (2026-03-07)
- [x] Phase 1-1 ~ 1-7 : 코드 기본틀 및 성능 최적화, UI 반영 완료 (2026-03-07)
- [x] Phase 1-8 : 하이브리드 진동 감지 로직 적용 및 검증용 스크립트 작성 완료 (2026-03-09)
