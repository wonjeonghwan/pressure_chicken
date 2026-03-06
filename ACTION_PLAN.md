# 압력밥솥 타이머 — 액션플랜

> 최종 업데이트: 2026-03-06 | 현재 단계: Phase 1-2 (auto_label.py)

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
   - 프롬프트 예시: `"pressure cooker"`, `"pot weight"`, `"burner"`
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

> `auto_label.py` — Step 1-3 에서 구현 예정

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

`core/state_machine.py` — 상태 6개로 확장

| 상태 | 설명 | UI 색상 |
|------|------|---------|
| `EMPTY` | 밥솥 없음 | 회색 |
| `POT_IDLE` | 밥솥 있음, 대기 | 파란색 |
| `POT_STEAMING_FIRST` | 초벌 12분 진행 중 | 초록색 |
| `DONE_FIRST` | 초벌 완료, 재벌 대기 | 노란색 |
| `POT_STEAMING_SECOND` | 재벌 5분 진행 중 | 진한 초록 |
| `DONE_SECOND` | 재벌 완료, 경보 | 빨간색 점멸 |

전환 규칙:
```
EMPTY → POT_IDLE          : pot_body 감지
POT_IDLE → STEAMING_FIRST : 딸랑이 진동 확정
STEAMING_FIRST → DONE_FIRST : 12분 완료
DONE_FIRST → STEAMING_SECOND : 딸랑이 재감지
STEAMING_SECOND → DONE_SECOND : 5분 완료

어느 상태 + pot 이탈 → EMPTY
STEAMING 중 → 자동 전환 없음 (잠금)
```

config 파라미터:
```json
"burners": [
  {
    "id": 1,
    "countdown_first":  720,
    "countdown_second": 300
  }
]
```

### Step 1-2. auto_label.py 구현

YOLO 모델로 이미지를 추론하여 YOLO 형식 라벨 파일 자동 생성.
사람이 검토 후 틀린 것만 수정하는 워크플로우 지원.

### Step 1-3. UI 업데이트

- 카드에 초벌/재벌 상태 구분 표시
- DONE_FIRST 상태에서 재벌 대기 안내 메시지
- 수동 조작: 초벌 시작 / 재벌 시작 / 초기화 버튼

### Step 1-4. config 유효성 검사

실행 시 config 오류를 사람이 읽기 쉬운 메시지로 출력.

---

## Phase 2 — 테스트 (카메라 1대)

### 환경
- 카메라 1대, 화구 5~10개
- 실제 압력밥솥 + 딸랑이

### 체크리스트

**감지 정확도**
- [ ] 빈 화구에서 EMPTY 유지
- [ ] 밥솥 올리면 POT_IDLE 전환
- [ ] 밥솥 내리면 EMPTY 복귀
- [ ] 딸랑이 2~3초 이상 → 초벌 12분 자동 시작
- [ ] 연기/증기 발생 시에도 오탐 없음
- [ ] 사람이 지나가며 가려도 타이머 유지

**사이클 전환**
- [ ] 12분 완료 → DONE_FIRST 전환
- [ ] DONE_FIRST 상태에서 딸랑이 재감지 → 5분 자동 시작
- [ ] 5분 완료 → DONE_SECOND 경보
- [ ] 밥솥 이탈 → EMPTY 리셋

**타이머 잠금**
- [ ] STEAMING 중 카메라 가림 후 딸랑이 재보여도 타이머 이중 시작 없음
- [ ] STEAMING 중 자동 상태 변경 없음

**수동 조작**
- [ ] 초벌 강제 시작
- [ ] 재벌 강제 시작
- [ ] 수동 초기화
- [ ] 화구 선택 (클릭 / 숫자키)

**다중 화구**
- [ ] 5개 화구 동시 실행 시 간섭 없음
- [ ] 각 화구 독립적으로 상태 유지

---

## Phase 3 — 확장 (카메라 2대)

### 체크리스트
- [ ] `config/store_config.json` 에 source 추가만으로 동작
- [ ] 두 카메라 화구가 독립적으로 동작
- [ ] 한 카메라 연결 끊겨도 나머지 정상 동작
- [ ] 화구 20개 동시 실행 성능 (30fps 유지)

---

## 미결 / 이후 고려사항

| 항목 | 내용 |
|------|------|
| 알람 | DONE_SECOND 시 소리 알람 추가 여부 |
| 로그 | 화구별 타이머 이력 저장 여부 |
| 원격 모니터링 | 다른 화면(태블릿 등)에서 확인 여부 |
| 새 매장 추가 | 카메라 배치가 다른 매장마다 재학습 필요 여부 |

---

## 현재 진행 위치

### Phase 0 — 데이터 준비

- [x] Phase 0-1 : 영상 확보 (주방 녹화 영상)
- [ ] Phase 0-2 : Roboflow 자동 라벨링 50장
- [ ] Phase 0-3 : 1차 학습 (부트스트랩 모델)
- [ ] Phase 0-4 : 나머지 프레임 자동 라벨링 + 검토
- [ ] Phase 0-5 : 최종 학습 (mAP@0.5 > 0.85 목표)

### Phase 1 — 코드 업데이트

- [x] Phase 1-1 : 상태머신 6상태 확장 (초벌/재벌 분리) — 2026-03-06 완료
- [ ] Phase 1-2 : `auto_label.py` 구현 (YOLO 추론 → YOLO 라벨 자동 생성)
- [ ] Phase 1-3 : UI 업데이트 (초벌/재벌 구분 표시, 재벌 대기 안내)
- [ ] Phase 1-4 : config 유효성 검사

### Phase 2 — 테스트 (카메라 1대)

- [ ] 감지 정확도 체크리스트 (GUIDE.md Phase 2 참고)
- [ ] 사이클 전환 체크리스트
- [ ] 타이머 잠금 체크리스트
- [ ] 수동 조작 체크리스트

### Phase 3 — 확장 (카메라 2대)

- [ ] 2대 동작 확인
- [ ] 20개 화구 동시 성능 (30fps 유지)
