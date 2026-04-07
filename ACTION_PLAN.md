# 압력밥솥 타이머 — 액션플랜

> 최종 업데이트: 2026-04-05 (6차) | 현재 단계: Phase 2 — OpticalFlow 방식 확정, 현장 테스트 대기

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

## Phase 0 — 데이터 준비 및 라벨링 ✅ 완료

- [x] 영상에서 프레임 추출 (`extract_frames.py`)
- [x] Roboflow 라벨링 (v3, 152장 증강 / 138 train + 14 val)
- [x] 1차 부트스트랩 학습
- [x] 최종 학습 완료 — mAP@0.5=95.3% (2026-03-07)

---

## Phase 1 — 코드 구현 ✅ 완료

### 상태머신 (core/state_machine.py) — 6상태

| 상태 | 설명 | UI 색상 |
|------|------|---------|
| `EMPTY` | 밥솥 없음 | 회색 |
| `POT_IDLE` | 밥솥 있음, 대기 | 파란색 |
| `POT_STEAMING_FIRST` | 초벌 12분 진행 중 | 초록색 |
| `DONE_FIRST` | 초벌 완료, 재벌 대기 | 노란색 |
| `POT_STEAMING_SECOND` | 재벌 5분 진행 중 | 진한 초록 |
| `DONE_SECOND` | 재벌 완료, 경보 | 빨간색 점멸 |

### 움직임 감지 (core/frame_processor.py) — 현재 MaskIoU 방식

**MaskIoUTracker** — 매 프레임마다:
1. YOLO-seg 결과에서 딸랑이 segmentation 폴리곤(`mask_xy`) 추출
2. 이전/현재 폴리곤을 **공통 캔버스**(두 bbox 합집합 + 50% 여백)에 64×64 binary mask로 렌더링
   - 캔버스는 절대 이미지 좌표 기준 → 위치 이동도 변화로 포착
3. `IoU = intersection / union` 계산
   - IoU ≈ 1.0 → 동일 영역 점유 → 정지
   - IoU 낮아짐 → 이전에 없던 영역 차지 → 움직임 (3D 회전 포함)
4. `IoU < iou_threshold` (기본값 0.75) → 해당 프레임 움직임으로 판정
5. 마스크 없는 프레임(연기·가림) → **skip** (window에 추가 안 함 → 타이머 유지)
6. 최근 `window_frames` 내 `trigger_frames` 개 이상 움직임 → 확정

**왜 NCC 대신 마스크 IoU인가:**

| 문제 | NCC (폐기) | MaskIoU (현재) |
|------|-----------|---------------|
| YOLO bbox jitter | 크롭 위치 흔들림 → FP | 마스크 모양 비교 → jitter 무관 |
| 연기·조명 변화 | 픽셀값 변화 → FP | 마스크 형태 안변함 → 무시 |
| 딸랑이 3D 원형 회전 | 픽셀 diff 작음 → FN | 윤곽선 변화 → IoU 하락 → 감지 |
| 마스크 없는 프레임 | False 누적 → FN | skip → 타이머 유지 |

**현재 파라미터 (config/store_config.json):**
```json
"motion": {
  "window_frames": 25,
  "trigger_frames": 15,
  "iou_threshold": 0.75
}
```

**전체 감지 흐름:**
```
YOLO-seg 배치 추론 (ROI 전체 1회 호출)
  ↓
밥솥 매칭: ROI 중심점 ↔ pot_body 중심점 거리 기반
  ↓
딸랑이 독점 매칭: x축 거리 기반 그리디 (중복 배정 방지)
  ↓
MaskIoUTracker.update(has_weight, w_box, mask_xy)
  마스크 있음 → IoU 비교 → window 추가
  마스크 없음 → skip (이전 마스크 유지)
  ↓
window_frames=25 중 trigger_frames=15 이상 motion → 움직임 확정
  ↓
상태머신 갱신 (EMPTY / POT_IDLE / STEAMING / DONE)
```

**완료 항목:**
- [x] 6상태 상태머신 + 타이머 잠금
- [x] ROI 캘리브레이션 (드래그 기반, 전체 화구 완료)
- [x] 배치 YOLO 추론 (15fps, 화구 수 무관)
- [x] 딸랑이 독점 할당 (x축 거리 기반 그리디)
- [x] HybridVibrationTracker → NCC → **MaskIoU 방식으로 교체** (2026-03-31)
- [x] body_ttl: 밥솥 미감지 시 15프레임 메모리 유지
- [x] 모델 없음 시 OpenCV 폴백 + 경고 배너
- [x] 현장 카메라 라이브 테스트 1회 수행 (2026-03-26 추정)
- [x] Segmentation 재라벨링 및 모델 학습 (yolov8n-seg, 2026-03-31)

---

## Phase 2 — 진동 감지 성능 튜닝 ✅ 완료 (OpticalFlow 방식 확정)

### 2026-04-02 ~ 04-05 변경 이력

**MaskIoUTracker → OpticalFlowDetector 전면 교체**

MaskIoU 방식의 근본적 한계 발견:
- YOLO seg 마스크가 프레임당 10~20%만 추출됨 (연기·조명에 따라 누락 심각)
- IoU 신호 자체가 너무 희박해 window 기반 판정이 불안정

→ Farneback Dense Optical Flow 기반 `OpticalFlowDetector`로 교체

**현재 구조 (2026-04-05 확정):**

```
영상 프레임
  ↓
Phase 1: Stabilizer (LK + RANSAC + EMA warpAffine) — 카메라 흔들림 제거
  ↓
YOLO-seg 배치 추론
  ↓
body/weight 매칭 (ROI 중심거리 기반 그리디)
  ↓
Phase 2: OpticalFlowDetector (Farneback dense flow)
  - weight bbox 내 전체 픽셀 이동량 RMS 계산
  - EMA 스무딩 (alpha=0.35) — 스파이크 억제
  - window=25 / trigger=14 투표 → 진동 확정
  ↓
상태머신 갱신 (EMPTY / POT_IDLE / STEAMING / DONE)
```

**Phase 3 (주파수 분석) 시도 및 결론:**

| 시도 | 신호 | 결과 | 원인 |
|------|------|------|------|
| 픽셀 밝기 FFT | bbox 평균 밝기 | 실패 | 주방 조명이 1~8Hz로 변동 → 오탐 |
| YOLO 중심점 FFT | 중심점 x좌표 | 실패 | YOLO 정수 좌표 → 0.5px 진폭으로 양자화 노이즈에 묻힘 |
| 전체 mean_flow_x FFT | bbox 평균 수평 flow | 실패 | 배경 픽셀이 방향 신호 희석 → 평균 ≈ 0 |
| 마스크 masked_flow_x | 마스크 픽셀 평균 수평 flow | 실패 | 마스크 추출 프레임 10~20% 이하, amp 평균 0.054 (임계값 0.3 미달) |

**최종 결론: Phase 3 비활성화, P2만으로 운영**
- Burner 9(진동): P2 motion 30.7% vs Burner 10(정지): 2.7% → 10배 차이 → P2 충분
- `config/frequency/enabled: false` 설정

**현재 파라미터 (store_config.json):**
```json
"optical_flow": {
  "rms_threshold": 0.5,
  "rms_ema_alpha": 0.35,
  "window_frames": 25,
  "trigger_frames": 14
},
"frequency": {
  "enabled": false
}
```

### 체크리스트
- [x] ROI 캘리브레이션 (전체 화구 완료)
- [x] HybridVibrationTracker → NCC → MaskIoU → **OpticalFlow로 교체** 완료
- [x] Phase 1 Stabilizer 통합 (core/stabilizer.py)
- [x] EMA 스무딩 튜닝 (오탐 57프레임 수준으로 억제)
- [x] Phase 3 주파수 분석 시도 및 실패 결론
- [x] Phase 3 비활성화 결정
- [ ] **현장 카메라 라이브 테스트** ← 다음 단계
- [ ] 딸랑이 움직임 → STEAMING 자동 전환 안정 확인
- [ ] 사이클 전환 체크리스트 (초벌 → 대기 → 재벌)
- [ ] 타이머 잠금 체크리스트 (사람 가림, 연기 발생 시)

### 현재 문제 분석

현장 라이브 테스트에서 **FP(오탐)와 FN(미탐) 동시 발생** 확인:

| 문제 | 증상 | 근본 원인 |
|------|------|-----------|
| **FP** (오탐) | 정지 딸랑이를 진동으로 판정 | YOLO 박스 jitter → `_POS_ALPHA=0.5` (빠른 EMA) → 크롭 영역이 매 프레임 1~2px 흔들림 → NCC 낮아짐 |
| **FN** (미탐) | 진동 딸랑이를 정지로 판정 | 박스 전체 크롭 → 배경 픽셀이 신호 희석 / `_MIN_STD` 미달 시 `motion=False` 누적 |

```
FP 경로:
  YOLO jitter → EMA(_POS_ALPHA=0.5 빠름) → 크롭 위치 흔들림
  → prev/curr 크롭이 1~2px 어긋남 → NCC < 0.85 → 오탐

FN 경로:
  딸랑이 박스 전체 크롭 → 주변 배경 포함 → 진동 신호 희석
  OR _MIN_STD 미달(어두운 환경) → NCC 계산 스킵 → motion=False 강제
```

### 성능 향상 접근 방안

#### 1. 즉시 시도 — 파라미터 조정

| 파라미터 | 현재값 | 조정 방향 | FP/FN 효과 |
|---------|--------|-----------|----------|
| `_POS_ALPHA` | 0.5 (코드 상수) | **0.2 로 낮추기** | FP↓ — 크롭 위치 흔들림 억제 |
| `ncc_threshold` | 0.85 | **0.90~0.92** | FP↓ — 미세 변화를 정지로 판정 |
| `_MIN_STD` | 3.0 (코드 상수) | **1.5 로 낮추기** | FN↓ — 어두운 환경에서 NCC 스킵 감소 |
| `_CROP_SIZE` | 32 | **48~64** | FN↓ — 더 많은 픽셀로 진동 패턴 캡처 |
| `window_frames` | 15 | **20~25** | FP↓ — 일시적 노이즈에 둔감 |
| `trigger_frames` | 6 | **비율 유지** (예: 8/20) | FP↓ FN↓ 균형 |

> **우선 조합 권장**: `_POS_ALPHA=0.2` + `ncc_threshold=0.92` + `_MIN_STD=1.5`
> 이 세 가지가 FP/FN 동시 문제의 핵심 원인에 직결됨

#### 2. 구조 개선 — 크롭 기준 고정 (앵커 크롭)

**현재**: 매 프레임 `prev → curr` 1-step 비교
- YOLO jitter가 있으면 크롭 위치도 jitter → NCC가 정지에서도 낮아짐

**개선**: POT_IDLE 진입 직후 N프레임 평균 크롭을 **"정지 기준 앵커"**로 저장
- 이후 모든 프레임을 이 앵커와 비교
- 효과: 1-step jitter 완전 제거, 정지 상태 NCC가 안정적으로 1.0에 수렴

```python
# 상태머신이 POT_IDLE 진입을 알려주면 tracker.set_anchor() 호출
# anchor는 진입 후 첫 5~10프레임 평균 크롭으로 확정
```

#### 3. 구조 개선 — _MIN_STD 처리 방식

**현재**: `_MIN_STD` 미달 시 `motion=False` → 윈도우에 False 누적 → FN 유발

**개선**: `_MIN_STD` 미달 시 윈도우에 **아무것도 append하지 않음** (skip)
- 어두운 환경에서 측정 불가 프레임을 "정지"로 오해하지 않음
- `deque(maxlen=window)` 대신 실질 유효 프레임 수 기준으로 판정

#### 4. 구조 개선 — 딸랑이 내부 크롭 비율 조정

**현재**: 박스 전체(100%) 크롭 → 배경 포함
**개선**: 박스 내부 **60~70%** 만 크롭 (중앙부)
- 배경 픽셀 제거 → 딸랑이 본체 패턴 집중
- 특히 딸랑이 상단 키포인트(kp_top) 기준으로 크롭하면 흔들리는 부분에 집중 가능
  - `keypoints` 데이터는 이미 감지되고 있음 (detector.py)

#### 5. 진단 먼저 — NCC 값 분포 측정

```bash
uv run python test_motion.py --source raw/Sample01.mp4 --burner 1
```

- **정지 구간 NCC 분포**: 0.85 아래로 얼마나 내려오는지 확인
- **진동 구간 NCC 분포**: 0.85 위에 얼마나 머무는지 확인
- 두 분포가 겹치는 구간 → threshold 설정의 한계점

### 테스트 시나리오

| 시나리오 | 기대 결과 | FP/FN |
|---------|---------|-------|
| 정지 딸랑이 3분 관찰 | STEAMING 전환 없음 | FP 검증 |
| 딸랑이 진동 시작 후 10초 이내 | STEAMING 전환 | FN 검증 |
| 딸랑이 미세 진동 (막 시작할 때) | STEAMING 전환 | FN 검증 (어려운 케이스) |
| 손으로 밥솥 5초 가림 후 제거 | STEAMING 유지 | 잠금 로직 |
| 수증기 자욱한 구간 | STEAMING 유지 | 잠금 로직 |
| 밥솥 이탈 후 재거치 | EMPTY → POT_IDLE | 상태 전환 |

---

## Phase 3 — 확장 (카메라 2대) ⏳ 대기

- [ ] 2대 동작 확인 (config 설정 완료)
- [ ] 20개 화구 동시 실행 성능 (15fps 유지 목표)

---

## 주요 설계 변경 이력

| 날짜 | 파일 | 변경 내용 |
|------|------|----------|
| 2026-03-06 | core/state_machine.py | 4상태 → 6상태 (STEAMING_FIRST/SECOND, DONE_FIRST/SECOND) |
| 2026-03-06 | calibration.py | 화구 수 고정 → 드래그 기반 동적 지정 방식으로 재작성 |
| 2026-03-07 | core/frame_processor.py | read_frames() / detect_and_update() 분리; 배치 YOLO 추론 |
| 2026-03-07 | main.py | 15fps 영상읽기 / 15fps YOLO; FPS 카운터 추가 |
| 2026-03-09 | core/frame_processor.py | `HybridVibrationTracker` 도입: EMA 보정, 템플릿 매칭 기반 진동 판별 |
| 2026-03-09 | test_motion.py | 구간별(Stationary/Moving) 정밀 분석용 스크립트 추가 |
| 2026-03-10 | ACTION_PLAN.md | 현재 상태 업데이트 및 Phase 1-8 추가 |
| 2026-03-11 | main.py, config | Mac/Windows 호환성 버그 수정 (`apply_source_overrides` 복구 및 웹캠 기본화) |
| ~2026-03-14 | core/frame_processor.py | EMA 가중치 조정, body_ttl 단축 (commit: 6f20cdf) |
| ~2026-03-26 | core/frame_processor.py | **`HybridVibrationTracker` → `FrameDiffTracker` (NCC 방식) 전면 교체** — 정지 딸랑이 오탐(FP) 문제로 인한 방식 변경 |
| ~2026-03-26 | main.py | 비디오 캡처 유틸 추가: 밝기/노출/감마 조정 테스트용 (commit: 2f2ad41) |
| 2026-03-27 | ACTION_PLAN.md | 현장 라이브 테스트 결과 반영, NCC 튜닝 방안 추가 |
| 2026-03-31 | dataset/ | Roboflow Segmentation 재라벨링 (129장) + augment_dataset.py 폴리곤 증강 (train 3090장) |
| 2026-03-31 | train.py | yolov8n-seg.pt 학습으로 전환, models/pot_seg.pt 출력 |
| 2026-03-31 | core/detector.py | Detection에 mask_xy 필드 추가, detect_batch()에서 r.masks.xy 추출 |
| 2026-03-31 | core/frame_processor.py | **`FrameDiffTracker`(NCC) → `MaskIoUTracker`(마스크 IoU) 전면 교체** — 3D 원형 회전 감지, 연기·조명 FP 해결 |
| 2026-04-02 | core/stabilizer.py | **Phase 1 Stabilizer 구현** — LK 특징점 추적 + RANSAC + EMA warpAffine 카메라 흔들림 보정 |
| 2026-04-02 | core/optical_flow.py | **Phase 2 OpticalFlowDetector 구현** — Farneback dense flow + EMA 스무딩(alpha=0.35) + window 투표 |
| 2026-04-03 | core/frame_processor.py | **MaskIoUTracker 완전 제거**, Phase 1+2 통합. full config dict로 생성자 변경 |
| 2026-04-03 | core/frequency_filter.py | **Phase 3 FrequencyAnalyzer 구현** — IIR bandpass(1~8Hz) + EMA amplitude |
| 2026-04-04 | core/optical_flow.py | mask_xy 파라미터 추가 → `last_masked_flow_x` 계산 지원 (Phase 3용) |
| 2026-04-05 | — | **Phase 3 실패 확인** — 4가지 신호 모두 실패. `frequency/enabled: false` 결정 |
| 2026-04-05 | tests/compare_phase3.py | Phase 3 시각 비교 뷰어 생성 (P2 RMS / P3 masked_flow_x / P3 Amp 실시간 그래프) |

---

## 현재 진행 위치 요약

```
Phase 0 ✅ → Phase 1 ✅ → Phase 2 ✅ (OpticalFlow 확정) → 현장 테스트 대기
```

**확정된 감지 스택:**
- Phase 1: Stabilizer (카메라 흔들림 보정)
- Phase 2: OpticalFlowDetector (RMS + EMA + window 투표)
- Phase 3: 비활성화 (주파수 분석 — 신호 품질 부족으로 포기)

**현재 파라미터:**
```json
"optical_flow": { "rms_threshold": 0.5, "rms_ema_alpha": 0.35, "window_frames": 25, "trigger_frames": 14 }
```

**핵심 미결 과제: 현장(실제 카메라) 테스트**
- 영상 파일 테스트 완료, 실제 주방 카메라 라이브 테스트 미완
- 오탐/미탐 케이스 수집 후 파라미터 재조정 예정
