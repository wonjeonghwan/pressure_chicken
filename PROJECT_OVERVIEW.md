# Pressure Chicken — AI 비전 기반 압력밥솥 타이머 시스템

> **한 줄 요약**: 음식점 주방의 여러 압력밥솥 딸랑이(추) 진동을 카메라로 자동 감지하고, 화구별 카운트다운 타이머를 관리하는 비전·AI 기반 모니터링 시스템.

---

## 목차

1. [프로젝트 배경 & 문제 정의](#1-프로젝트-배경--문제-정의)
2. [시스템 전체 파이프라인](#2-시스템-전체-파이프라인)
3. [Phase 1 — Stabilizer (카메라 흔들림 보정)](#3-phase-1--stabilizer)
4. [YOLO-seg 객체 감지](#4-yolo-seg-객체-감지)
5. [화구·딸랑이 매칭 로직](#5-화구딸랑이-매칭-로직)
6. [Phase 2 — Optical Flow 진동 감지](#6-phase-2--optical-flow-진동-감지)
7. [7상태 유한 상태머신 (FSM)](#7-7상태-유한-상태머신-fsm)
8. [YOLO 데이터셋 & 모델 학습](#8-yolo-데이터셋--모델-학습)
9. [설정 시스템 (store_config.json)](#9-설정-시스템-store_configjson)
10. [UI/UX 설계 — pygame 통합 대시보드](#10-uiux-설계--pygame-통합-대시보드)
11. [다카메라 확장 설계](#11-다카메라-확장-설계)
12. [실패 기록 — Phase 3 주파수 분석 & MaskIoU](#12-실패-기록--phase-3-주파수-분석--maskiou)
13. [설계 변경 전체 이력](#13-설계-변경-전체-이력)
14. [개발 Phase별 완료 이력](#14-개발-phase별-완료-이력)
15. [매장 셋업 & 운영 가이드](#15-매장-셋업--운영-가이드)
16. [프로젝트 파일 구조](#16-프로젝트-파일-구조)
17. [의존성 & 실행 방법](#17-의존성--실행-방법)
18. [현재 성능 지표 & 파라미터](#18-현재-성능-지표--파라미터)
19. [미결 과제 & 로드맵](#19-미결-과제--로드맵)

---

## 1. 프로젝트 배경 & 문제 정의

### 해결하려는 문제

음식점(누룽지 삼계탕 전문점 등) 주방에서는 여러 압력밥솥을 동시에 운영한다. 각 밥솥이 압력에 도달하면 뚜껑 위 딸랑이(추)가 흔들리기 시작하는데, 이 시점부터 조리 타이머를 카운트다운해야 한다.

**기존 방식의 문제점**:
- 주방 직원이 밥솥 딸랑이를 직접 눈으로 보며 타이머를 수동으로 시작해야 함
- 여러 화구를 동시에 봐야 하므로 놓치기 쉬움
- 조리 과정: **초벌 12분 → 냉각 10분 → 재벌 5분** 두 단계를 모두 관리해야 함

### 솔루션

천장 또는 벽에 설치한 USB 카메라(1~5대 이상)로 화구들을 촬영하고, AI 비전 파이프라인이:

1. YOLO segmentation 모델로 밥솥 몸체(`pot_body`)와 딸랑이(`pot_weight`)를 감지
2. Optical Flow로 딸랑이의 진동을 계산
3. 진동이 확정되면 자동으로 카운트다운 시작
4. 7단계 상태머신으로 초벌→완료→재벌→완료 사이클을 자동 관리

**핵심 가치**:
- 완전 로컬 동작 — 학습 시 모델 1회 다운로드 이후 인터넷 불필요
- 카메라 1대로 여러 화구 동시 감지, 또는 카메라 N대로 더 넓은 주방 커버
- 잘못된 감지(손, 연기, 일시적 가림)에도 타이머가 날아가지 않는 잠금 로직

---

## 2. 시스템 전체 파이프라인

```
영상 프레임 입력 (카메라 N대, 각 15 또는 10fps)
      │
      ▼
Phase 1: Stabilizer (core/stabilizer.py)
  - Shi-Tomasi 코너 특징점 검출 → LK 광학흐름 추적
  - RANSAC으로 배경 인라이어(공통 이동) 선별
  - EMA 스무딩한 역방향 warpAffine 보정
  → 카메라 흔들림 제거된 안정 프레임
      │
      ▼
YOLO-seg 배치 추론 (core/detector.py)
  - 화구 ROI 합집합을 단일 crop으로 1회 호출 (N화구라도 1회)
  - 동적 margin = max(30, min(50, 짧은변 × 10%))
  - 클래스: empty_burner(0) / pot_body(1) / pot_weight(2)
  - mask_xy(segmentation 폴리곤) 포함 반환
      │
      ▼
화구·딸랑이 매칭 (core/frame_processor.py)
  - ROI 중심거리 기반 greedy 매칭
  - body_ttl: 밥솥 일시 가림 15프레임 메모리 유지
  - 딸랑이 독점 할당 (x축 거리 기반 greedy, 중복 없음)
      │
      ▼
Phase 2: OpticalFlowDetector (core/optical_flow.py)
  - Farneback dense optical flow 계산
  - crop 위치: bbox center EMA (pos_alpha=0.3) — mask 유무 무관
  - RMS 계산: mask_xy 폴리곤 내부 픽셀만 (없으면 bbox 전체 fallback)
  - 평균 flow 차감 → residual RMS (순수 형상 변화만)
  - RMS 정규화: norm_rms = raw_rms × ref_diag / bbox_diag
  - EMA 스무딩 (alpha=0.35) → window 투표 (25프레임, 14개 이상 → 진동 확정)
      │
      ▼
상태머신 갱신 (core/state_machine.py)
  - BurnerStateMachine.update(pot_present, vibrating)
  - 7상태 전환 + 타이머 잠금 + pot_absent debounce
      │
      ▼
UI 렌더링 (ui/ui_display.py)
  - pygame 통합 대시보드
  - 좌: 카메라 그리드 (N대 균등 분할) + YOLO 오버레이
  - 우: 화구 카드 목록 (상태·카운트다운·진동게이지)
  - 인앱 캘리브레이션 (F키 → 드래그 → Enter 저장)
```

---

## 3. Phase 1 — Stabilizer

**파일**: `core/stabilizer.py`

카메라가 실제로 흔들리거나 조명·환경에 따라 미세 진동이 있으면, Optical Flow가 배경 움직임을 딸랑이 진동으로 오인(FP)한다. Phase 1은 이를 막기 위해 각 프레임에서 카메라 자체의 이동량을 추정하고 역방향으로 보정한다.

### 알고리즘 흐름

```
1. goodFeaturesToTrack (Shi-Tomasi 코너)로 추적 특징점 동적 선별
   - textured 영역 위주 → 밋밋한 배경 배제 → LK 추적 정확도 향상
   - 특징점 부족 시 (min_inliers 미만) 균등 Grid로 fallback

2. calcOpticalFlowPyrLK 로 이전→현재 프레임 간 특징점 추적

3. estimateAffinePartial2D + RANSAC
   - 배경 인라이어(공통 이동) 선별 → 전경 이상치 제거
   - 결과: 순수 카메라 이동량 (dx, dy)

4. EMA 스무딩
   smooth_dx = alpha * raw_dx + (1 - alpha) * smooth_dx
   → 급격한 튐 억제, 서서히 보정 적용

5. 역방향 warpAffine
   correction = [[1, 0, -smooth_dx], [0, 1, -smooth_dy]]
   → BORDER_REPLICATE로 경계 채움
```

### 설계 원칙

- **프레임 간(frame-to-frame) 보정** — 누적 변환(cumulative) 대신 사용. 오차 누적(drift) 없음.
- **고주파 떨림만 제거, 저주파 이동은 통과** — `smooth_alpha`로 제어 (낮으면 느린 떨림도 제거, 높으면 빠른 떨림만 제거).
- **소스별 독립 인스턴스** — 카메라마다 별도 Stabilizer. 한 카메라의 흔들림이 다른 카메라에 영향 없음.

### 파라미터 (store_config.json "stabilizer")

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `enabled` | `true` | Stabilizer 활성화 여부 |
| `grid_rows` / `grid_cols` | 6 / 8 | Fallback grid 격자 수 |
| `lk_win_size` | 21 | LK 추적 윈도우 크기(px) |
| `lk_max_level` | 3 | LK 피라미드 레벨 |
| `ransac_threshold` | 3.0 | RANSAC 인라이어 판정 거리(px) |
| `smooth_alpha` | 0.3 | EMA 계수 (낮을수록 부드럽게) |
| `min_inliers` | 6 | 최소 인라이어 수 (미달 시 보정 스킵) |

---

## 4. YOLO-seg 객체 감지

**파일**: `core/detector.py`

### 감지 클래스 (3가지)

```yaml
# dataset/dataset.yaml
nc: 3
names:
  0: empty_burner   # 빈 화구 (밥솥 없음)
  1: pot_body       # 밥솥 몸체 (기준점 역할)
  2: pot_weight     # 딸랑이 (추)
```

### 배치 추론 구조

화구가 여러 개라도 YOLO는 카메라당 **1회만** 호출한다.

```python
# 화구들의 ROI 합집합을 하나의 crop으로 만들어 단일 호출
margin = max(30, min(50, int(min_short_side * 0.1)))  # 동적 margin
crop = stabilized[cy1:cy2, cx1:cx2]
dets = detector.detect(crop)
# 좌표를 원본 이미지 좌표계로 복원
for d in dets:
    d.x1 += cx1; d.x2 += cx1
    d.y1 += cy1; d.y2 += cy1
```

**동적 margin**: ROI 짧은 변의 10%, 최소 30px, 최대 50px.
- 작은 ROI: 작은 margin → YOLO 입력 면적 절감
- 큰 ROI: 큰 margin → stabilizer 보정 후 객체 이탈 방지

### Detection 데이터 클래스

```python
@dataclass
class Detection:
    class_id:   int           # 0/1/2
    confidence: float
    x1: int; y1: int; x2: int; y2: int
    keypoints: list | None    # pose 모델용 (seg에서는 None)
    mask_xy: np.ndarray | None  # shape (N, 2) seg 폴리곤, 이미지 좌표계
```

### 모델 정보

- 베이스 모델: `yolov8n-seg.pt` (YOLOv8 nano segmentation)
- 학습된 모델: `models/pot_seg.pt` (6.5MB, 3.26M 파라미터)
- FP16(half): CUDA 사용 가능 시 자동 활성화
- 모델 파일 없을 경우: `model_missing=True` 플래그 → 빈 리스트 반환 (graceful degradation)

---

## 5. 화구·딸랑이 매칭 로직

**파일**: `core/frame_processor.py`

### 화구 ↔ 밥솥 몸체 매칭

캘리브레이션에서 설정한 ROI 기준으로, 그 영역 안에 중심이 들어오는 `pot_body` 감지 중 가장 가까운 것을 그 화구의 밥솥으로 매칭.

```python
for b in bodies:
    # ROI 20% 확장 영역 안에 중심이 들어오면 후보
    if (rx - mx <= b.cx <= rx + rw + mx) and (ry - my <= b.cy <= ry + rh + my):
        dist = math.hypot(b.cx - roi_cx, b.cy - roi_cy)
        if dist < best_dist:
            best_dist, best_body = dist, b
```

### 딸랑이 독점 매칭 (greedy)

한 딸랑이가 여러 화구에 동시 할당되지 않도록, x축 거리 기준으로 가까운 쌍부터 독점 매칭.

```python
candidates = []
for bid, body_box in matched_bodies.items():
    for wi, w in enumerate(weights):
        # 밥솥 bbox 15% 확장 영역 안의 딸랑이만 후보
        if in_range(body_box, w):
            candidates.append((abs(w.cx - body_cx), bid, wi))

candidates.sort(key=lambda t: t[0])  # 가까운 순
used_weights, used_bids = set(), set()
for _, bid, wi in candidates:
    if bid in used_bids or wi in used_weights:
        continue
    # 최근접 1:1 매칭
    matched_has_weight[bid] = (True, weight_box, mask_xy)
    used_weights.add(wi); used_bids.add(bid)
```

### body_ttl — 일시 가림 내성

밥솥이 감지되다가 갑자기 안 보여도 15프레임(~1초) 동안 "마지막 위치에 있다"고 간주. 손이나 수증기로 순간 가려져도 타이머가 날아가지 않는다.

```python
if self._body_ttl.get(bid, 0) > 0 and bid in self.last_matched_boxes:
    self._body_ttl[bid] -= 1
    detections[bid] = (True, vibrating)  # pot_present=True 유지
```

---

## 6. Phase 2 — Optical Flow 진동 감지

**파일**: `core/optical_flow.py`

딸랑이 bbox 영역에 Farneback dense optical flow를 계산하고, 잔차 RMS로 진동 여부를 판별한다.

### 전체 처리 흐름

```
1. Box jump 감지 (다른 물체로 교체됐는지 확인)
   - 이전/현재 bbox 중심 거리가 bbox 대각선의 50% 초과 → jump로 판정
   - jump: centroid EMA 리셋, history 클리어 (선택적)

2. Centroid EMA → 안정된 crop 위치
   - 항상 bbox center를 EMA에 반영 (mask 유무 무관)
   - pos_alpha=0.3: 이전 위치를 70% 유지 → YOLO jitter 완화
   - mask centroid를 crop 기준으로 쓰면 mask 없는 프레임마다 bbox center로
     끌려 oscillation → FP 폭증 (2026-04-19 폐기)

3. Farneback Dense Optical Flow
   flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, ...)
   파라미터: pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5

4. RMS 계산 (residual, deformation 측정)
   - mask_xy 폴리곤 내부 픽셀만 추출
   - 평균 flow를 차감 → 카메라 이동/딸랑이 위치 이동 제거
   - sqrt(mean((fx - mean_fx)² + (fy - mean_fy)²))
   - mask 없는 프레임: bbox 전체로 fallback

5. 크기 정규화
   norm_rms = raw_rms × ref_diag / bbox_diag
   - bbox가 크면(카메라 가까울수록) 나눠서 줄임
   - ref_diag로 다시 곱해 스케일 유지 (값이 0으로 수렴하지 않음)
   - 해상도·줌이 달라도 threshold 동일 적용 가능
   - ref_diag=40px: 현재 영상 기준 평균 딸랑이 bbox 대각선

6. EMA 스무딩
   ema_rms = alpha * norm_rms + (1 - alpha) * ema_rms
   alpha=0.35

7. Window 투표
   - deque(maxlen=25)에 (ema_rms > threshold) bool 저장
   - window가 25프레임 꽉 찼을 때 14개 이상 → 진동 확정 (VIBRATING)
   - window가 덜 찼으면 False (초기화 후 너무 빨리 판정 방지)
```

### 상태 전환별 OpticalFlow 리셋 정책

| 전환 | 리셋 이유 |
|------|---------|
| `DONE_FIRST` 진입 (초벌 완료) | 이전 STEAMING 구간 누적 flow 제거 |
| `WAIT_SECOND` 진입 | DONE_FIRST 구간 flow 히스토리 제거 |
| `EMPTY` 전환 | 이전 사이클 흔적 완전 초기화 |

### 핵심 파라미터 (현재 확정값)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `rms_threshold` | 0.20 | 진동 인정 임계값 (정지 noise p90=0.19 바로 위) |
| `rms_ema_alpha` | 0.35 | RMS EMA 스무딩 계수 |
| `pos_ema_alpha` | 0.3 | bbox centroid EMA 계수 |
| `normalize_rms` | `true` | bbox 크기 기준 정규화 활성 |
| `normalize_ref_diag` | 40.0 | 정규화 기준 대각선(px) |
| `window_frames` | 25 | 투표 window 길이 |
| `trigger_frames` | 14 | window 내 진동 프레임 수 임계값 |

**noise floor 실측값 (500프레임 정지 영상)**:
- norm_rms 범위: 0.015 ~ 0.13, p99 ≈ 0.32
- threshold 0.20 = noise p90(0.19) 바로 위
- window 투표(14/25)로 FP 추가 차단

---

## 7. 7상태 유한 상태머신 (FSM)

**파일**: `core/state_machine.py`

### 상태 정의

| 상태 | 설명 | 진동 감지 | UI 색상 (RGB) |
|------|------|-----------|---------------|
| `EMPTY` | 빈 화구 (밥솥 미감지) | — | 회색 `(80,80,80)` |
| `POT_IDLE` | 밥솥 감지, 대기 중 | 활성 | 브랜드 옐로우 `(255,192,0)` |
| `POT_STEAMING_FIRST` | 초벌 12분 타이머 진행 중 | 🔒 잠금 | 초록 `(60,180,60)` |
| `DONE_FIRST` | 초벌 완료, 냉각 중 | ❌ 무시 | 오렌지 `(255,140,0)` |
| `WAIT_SECOND` | 재벌 대기 (진동 재활성) | 활성 | 파랑 `(100,160,220)` |
| `POT_STEAMING_SECOND` | 재벌 5분 타이머 진행 중 | 🔒 잠금 | 진초록 `(30,130,30)` |
| `DONE_SECOND` | 재벌 완료, 경보 (빨간색 점멸) | — | 빨강 `(220,40,40)` |

### 전환 규칙

```
EMPTY → POT_IDLE              : pot_body 감지
POT_IDLE → STEAMING_FIRST     : 딸랑이 진동 확정 (window 투표 통과)
STEAMING_FIRST → DONE_FIRST   : 초벌 타이머 완료 (기본 12분)
DONE_FIRST → WAIT_SECOND      : done_first_timeout 경과 + pot 존재 (기본 10분)
DONE_FIRST → EMPTY            : pot 이탈 (debounce 적용)
WAIT_SECOND → STEAMING_SECOND : 딸랑이 진동 확정
WAIT_SECOND → EMPTY           : pot 이탈 (debounce 적용)
STEAMING_SECOND → DONE_SECOND : 재벌 타이머 완료 (기본 5분)
DONE_SECOND → EMPTY           : pot 이탈 (최종 상태, debounce 적용)
```

### 타이머 잠금 설계

`POT_STEAMING_FIRST` / `POT_STEAMING_SECOND` 상태에서는 카메라 감지 결과로 상태가 자동 변경되지 않는다. 딸랑이가 가려졌다가 다시 보여도 "또 시작"으로 오인하지 않는다. **수동 조작만** 상태를 바꿀 수 있다.

```python
elif self.state == BurnerState.POT_STEAMING_FIRST:
    if self.remaining_seconds <= 0:
        # 오직 타이머 완료 조건만 체크 — 진동/pot 여부 무시
        self._done_time      = time.monotonic()
        self._done_first_end = time.monotonic() + self._done_first_timeout
        self.state           = BurnerState.DONE_FIRST
```

### DONE_FIRST 냉각 단계 설계 이유

초벌 완료 직후에는 딸랑이에 잔여 진동이 남아 있어, 곧바로 "재벌 시작"으로 오인할 수 있다. 그래서 완료 후 `done_first_timeout`(기본 10분) 동안은 진동을 무시(`DONE_FIRST`)하고, 시간이 지나야 `WAIT_SECOND`로 넘어가 진동 감지를 재활성한다.

`status_label`은 DONE_FIRST 상태에서 냉각 남은 시간(`MM:SS`)을 카드에 표시한다.

### pot_absent_threshold — debounce 설계

완료 계열 상태(`DONE_FIRST` / `WAIT_SECOND` / `DONE_SECOND`)에서 밥솥을 **연속 N프레임(기본 30, ~30fps 기준 약 1초)** 못 봐야 "치워짐 → `EMPTY`"로 확정한다.

김·손·행주에 순간 가려도 타이머를 날리지 않기 위한 debounce 값. `EMPTY` 전환 시 내부 타이머 상태도 `_reset_timer()`로 함께 초기화.

```python
elif self.state == BurnerState.DONE_FIRST:
    if not pot_present:
        self._pot_absent_count += 1
        if self._pot_absent_count >= self._pot_absent_threshold:  # 30프레임
            self._reset_timer()
            self.state = BurnerState.EMPTY
    else:
        self._pot_absent_count = 0
        if self._done_first_end is not None and time.monotonic() >= self._done_first_end:
            self._done_first_end = None
            self.state = BurnerState.WAIT_SECOND
```

### 수동 조작 API

```python
def manual_reset(self) -> None:
    """수동 초기화 → EMPTY (1초 길게 누르기 방지 적용)"""
    self._reset_timer()
    self.state = BurnerState.EMPTY

def manual_start(self) -> None:
    """수동 타이머 강제 시작/진행"""
    if self.state in (BurnerState.EMPTY, BurnerState.POT_IDLE):
        self._start_first()           # 초벌 시작
    elif self.state == BurnerState.POT_STEAMING_FIRST:
        self._countdown_end = time.monotonic()  # 초벌 즉시 완료
    elif self.state == BurnerState.DONE_FIRST:
        self._done_first_end = None
        self._start_second()          # '바로 재벌' — 10분 냉각 스킵
    elif self.state == BurnerState.WAIT_SECOND:
        self._start_second()          # 재벌 강제 시작
    elif self.state == BurnerState.POT_STEAMING_SECOND:
        self._countdown_end = time.monotonic()  # 재벌 즉시 완료
```

### 조회 프로퍼티 (UI 유동 정렬용)

```python
@property
def is_counting(self) -> bool:
    """카운트다운 진행 중(초벌·재벌) — UI 유동 정렬 1순위 판정용"""
    return self.state in (POT_STEAMING_FIRST, POT_STEAMING_SECOND)

@property
def remaining_seconds(self) -> float:
    """남은 초. 카운트다운 중이 아니면 0.0"""
    ...

@property
def seconds_since_done(self) -> float:
    """카운트다운 종료 후 경과 초. 완료 이전이면 -1.0
    UI에서 '완료 직후 N초 상단 강조 유지' 판정에 사용"""
    ...
```

### BurnerRegistry

매장 전체 화구를 관리하는 컨테이너.

```python
class BurnerRegistry:
    def add(self, burner_id, countdown_first, countdown_second,
            done_first_timeout=600, pot_absent_threshold=30) -> BurnerStateMachine:
        ...
    def update_all(self, detections: dict[int, tuple[bool, bool]]) -> None:
        """detections: {burner_id: (pot_present, vibrating)}"""
        ...
```

---

## 8. YOLO 데이터셋 & 모델 학습

### 데이터셋 구성

- 라벨링 도구: Roboflow (segmentation 폴리곤)
- 최초 라벨링: 152장 (부트스트랩)
- 재라벨링 (2026-03-31): segmentation 129장 → 증강 후 train 3,090장

### 증강 파이프라인 (`augment_dataset.py`)

```python
# 2026-04-12 확정 파이프라인
augmentations = [
    A.Rotate(limit=40, p=0.8),          # RandomRotate90 대신 ±40° 연속 회전
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.3),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.2),
    A.HorizontalFlip(p=0.5),
]
# valid: 원본 복사만 (증강 없음)
```

### 학습 스크립트 (`train.py`)

```python
model = YOLO("yolov8n-seg.pt")  # 베이스 모델 (최초 1회 다운로드)
model.train(
    data="dataset/dataset.yaml",
    epochs=100,
    imgsz=640,
    close_mosaic=30,    # 마지막 30 epoch에서 mosaic 비활성 (2026-04-12 변경)
    workers=4,          # 발열 완화
    device=0,           # GPU
)
```

### 현재 모델 성능 (2026-04-12 기준, 87 epoch, best=71)

| 지표 | 이전 | 현재 |
|------|------|------|
| mAP50(B) | 75.4% | **85.0%** |
| mAP50(M) | 70.8% | **81.0%** |
| mAP50-95(B) | — | 67.8% |
| mAP50-95(M) | — | 54.4% |

---

## 9. 설정 시스템 (store_config.json)

### 3계층 합성 우선순위

```
global < source.optical_flow < burner.optical_flow
```

각 계층에서 override하고 싶은 키만 명시. 뒤 계층이 앞 계층 값을 덮어씀.

### 전체 구조 (주석 포함 설명)

```json
{
  "store_id": "store_001",
  "store_name": "압구정점",
  "sources": [
    {
      "id": 0,
      "type": "camera",          // "camera" 또는 "file" (테스트용)
      "index": 0,                // OS 카메라 인덱스
      "label": "Cam-0",          // UI 표시용 이름
      "exposure": -5,            // 자동 노출 OFF, 수동 고정 (-3 ~ -7)
      "resize": [640, 480],      // 강제 다운스케일 (선택)
      // source 단위 override:
      "optical_flow": { "rms_threshold": 0.25 },
      "stabilizer": { "smooth_alpha": 0.5 }
    }
  ],
  "stabilizer": {               // global stabilizer 설정
    "enabled": true,
    "grid_rows": 6, "grid_cols": 8,
    "lk_win_size": 21, "lk_max_level": 3,
    "ransac_threshold": 3.0,
    "smooth_alpha": 0.3,
    "min_inliers": 6
  },
  "optical_flow": {             // global optical flow 설정
    "enabled": true,
    "farneback_pyr_scale": 0.5,
    "farneback_levels": 3,
    "farneback_winsize": 15,
    "farneback_iterations": 3,
    "farneback_poly_n": 5,
    "farneback_poly_sigma": 1.2,
    "rms_threshold": 0.20,      // 진동 인정 임계값
    "rms_ema_alpha": 0.35,
    "pos_ema_alpha": 0.3,
    "window_frames": 25,        // 투표 window 길이
    "trigger_frames": 14,       // window 내 진동 프레임 수 임계값
    "normalize_rms": true,
    "normalize_ref_diag": 40.0,
    "max_box_jump_ratio": 0.5,
    "reset_on_box_jump": true,
    "reset_on_missing_box": false,
    "missing_reset_frames": 3
  },
  "model": {
    "weights": "models/pot_seg.pt",
    "confidence": 0.4           // YOLO 감지 신뢰도 임계값
  },
  "burners": [
    {
      "id": 1,
      "source_id": 0,           // 어느 카메라가 이 화구를 담당하는지
      "roi": [200, 200, 400, 400],  // [x, y, w, h] — F키 캘리브로 자동 설정
      "countdown_first": 720,   // 초벌 타이머 초 (기본 12분)
      "countdown_second": 300,  // 재벌 타이머 초 (기본 5분)
      "done_first_timeout": 600,// 냉각 대기 초 (기본 10분)
      "pot_absent_threshold": 30, // 이탈 판정 연속 미감지 프레임 수
      // burner 단위 override:
      "optical_flow": { "rms_threshold": 0.18 }
    }
  ],
  "ui": {
    "grid_cols": 6,
    "window_size": [1280, 720],
    "window_title": "압력밥솥 타이머"
  },
  "frequency": {
    "enabled": false            // Phase 3 주파수 분석 — 실패로 비활성화
  }
}
```

### 예시 config 파일

| 파일 | 용도 |
|------|------|
| `config/store_config.json` | 현재 매장 운영 config |
| `config/examples/store_1cam.json` | 카메라 1대 매장 |
| `config/examples/store_3cam.json` | 카메라 3대 매장 |
| `config/examples/store_4cam.json` | 카메라 4대 매장 (현재 기준) |
| `config/examples/store_5cam.json` | 카메라 5대 매장 |
| `config/examples/store_4cam_video.json` | 4영상 시뮬레이션 (테스트용) |

---

## 10. UI/UX 설계 — pygame 통합 대시보드

**파일**: `ui/ui_display.py`

### 화면 레이아웃

```
┌──────────────────────────────────────┬──────────────────────┐
│  카메라 그리드 (좌측, N대 균등 분할)  │  우측 패널 (400px)    │
│  ┌──────────┬──────────┐             │  ─────────────────── │
│  │ [Cam-0] ●│ [Cam-1] ●│             │  ⚙설정  Mask  Dev    │
│  │  영상+    │  영상+    │             │  ─────────────────── │
│  │  오버레이 │  오버레이 │             │  ┌──────┐ ┌──────┐  │
│  ├──────────┼──────────┤             │  │ #1   │ │ #2   │  │
│  │ [Cam-2] ●│ [Cam-3] ●│             │  │ 초벌  │ │ 재벌  │  │
│  │  영상+    │  영상+    │             │  │ 3:42 │ │ 1:15 │  │
│  │  오버레이 │  오버레이 │             │  │ ⟳ ▶ │ │ ⟳ ▶ │  │
│  └──────────┴──────────┘             │  └──────┘ └──────┘  │
└──────────────────────────────────────┴──────────────────────┘
```

- **좌측**: 카메라 그리드. N대에 따라 1×1 / 1×2 / 2×2 / 2×3 자동 결정.
- **우측 패널**: 툴바 + 화구 카드 목록. 각 카드에 화구 번호, 상태, 카운트다운, 진동 게이지.

### 카메라 그리드 자동 레이아웃

```python
def _grid_layout(n_cams: int) -> tuple[int, int]:
    if n_cams <= 1: return (1, 1)
    if n_cams == 2: return (1, 2)
    if n_cams <= 4: return (2, 2)
    if n_cams <= 6: return (2, 3)
    if n_cams <= 9: return (3, 3)
    return (3, 3)  # 10+ → 페이지네이션
```

### 단축키 & 조작

| 키 / 버튼 | 동작 |
|----------|------|
| `F` | 캘리브레이션 모드 진입 |
| 캘리브 `Enter` | ROI 저장 후 핫 리로드 |
| 캘리브 `Z` | 마지막 ROI 취소 |
| 캘리브 `Del` / `Backspace` | 선택 화구 삭제 |
| 캘리브 `[` / `]` | 같은 카메라 안에서 화구 ID 한 칸 이동 |
| 캘리브 `Esc` | 저장 없이 취소 |
| `1`~`9`, `0` | 화구 1~10번 직접 선택 |
| `R` (1초 길게) | 선택 화구 리셋 → EMPTY |
| `S` | 선택 화구 타이머 강제 시작 |
| `M` | mask 오버레이 토글 |
| `D` | 개발 모드 (RMS 수치·번호 크게 등) 토글 |
| `Space` | 비디오 일시정지 (파일 소스 시) |
| `C` | 카메라 전환 |
| `Q` | 종료 |

### 인앱 캘리브레이션 (F키)

별도 스크립트 없이 메인 UI 내에서 바로 실행. 캘리브레이션 중에도 카메라 영상이 실시간으로 보이므로 화구 위치를 정확하게 지정할 수 있다.

```
F 키 입력 → 캘리브 모드
  ↓
영상 위에서 마우스 드래그 → ROI 박스 그리기
  ↓
클릭한 셀의 source_id가 자동 귀속
  ↓
Enter → 카메라별 그룹화 후 ID 1부터 자동 재부여 → config 저장 → 핫 리로드
```

**화구 ID 자동 그룹화 (저장 시)**:
```
그릴 때:                         저장 후:
  Cam-0[A, B]                      Cam-0 → ID 1, 2
  Cam-2[C]       → 자동 그룹화 →   Cam-1 → ID 3
  Cam-1[D]                         Cam-2 → ID 4
  Cam-3[E]                         Cam-3 → ID 5
```

### 우측 패널 버튼

| 버튼 | 동작 |
|------|------|
| `⚙ 설정(F)` | 캘리브레이션 모드 진입 |
| `Mask(M)` | segmentation 마스크 오버레이 토글 |
| `Dev(D)` | 개발 모드 (RMS·점수 등 표시) |
| `+ 추가` | 빈 카메라 인덱스 자동 탐색 후 추가 |
| `− 제거` | 선택된 카메라 제거 (묶인 화구도 함께) |

### 처리 비용 진단 로그 (실행 시 자동 출력)

```
[main] target_fps=10.0 (cams=4)
[diag] Cam-0: burners=2, ROI 합집합=34.7%  → ✓ 가벼움
[diag] Cam-1: burners=2, ROI 합집합=34.7%  → ✓ 가벼움
[diag] Cam-2: burners=4, ROI 합집합=78.3%  → ⚠⚠ 카메라 추가 검토
[diag] Cam-3: 화구 미할당
```

| ROI 합집합 비율 | 판정 |
|---------------|------|
| 40% 미만 | 🟢 가벼움 (안정 운영) |
| 40~65% | 🟡 적정 |
| 65% 이상 | 🔴 카메라 추가 분담 검토 |

### 알림음 시스템

주방 소음 대응: 2·3배음 혼합 + sustain 엔벨로프 + 음량 0.9. 외부 음원·인터넷 불필요. `make_sounds.py`로 WAV 합성.

| 파일 | 재생 시점 | 설명 |
|------|---------|------|
| `assets/sounds/warn_30s.wav` | 초벌·재벌 완료 30초 전 | 2회 비프 + 상승음 (~0.66초) |
| `assets/sounds/complete.wav` | 초벌·재벌 완료 순간 | 4음 상승 팡파레 (~1.03초) |
| `assets/sounds/cam_lost.wav` | 카메라 연결 끊김 감지 | 낮은 음 3회 반복 (~1.1초) |

---

## 11. 다카메라 확장 설계

**파일**: `MULTI_CAMERA_PLAN.md`

### 목표 조건

- 카메라 N대(1~8대+) 동일 코드로 동작. 하드코딩된 매직 넘버 금지.
- A매장 3대 / B매장 4대 / C매장 5대 — 동일 코드 + config만으로 대응.
- 한 매장 안에서 카메라 모델·해상도 혼합 허용 (1080p 1대 + 480p 3대 등).
- 저사양 노트북에서 **최소 10fps 유지**.
- 한 카메라가 끊겨도 다른 화구 타이머는 영향 없음 (격리).

### 카메라 수별 권장 설정

| 카메라 수 | target fps | 권장 해상도 | 비고 |
|----------|-----------|------------|------|
| 1~2대 | 15 | 720p~1080p | 현재 단독 운영 형태 |
| 3~4대 | 10 | 720p | 현재 매장 기준 |
| 5~6대 | 8~10 | 720p | USB 허브 분산 필수 |
| 7대+ | 8 | 720p | 외부 GPU 또는 미니PC 권장 |

### target_fps 자동 결정

```python
n_cams = len(sources)
if n_cams <= 2:   target_fps = 15
elif n_cams <= 6: target_fps = 10
else:             target_fps = 8
```

### 구현 로드맵

| Phase | 내용 | 상태 |
|-------|------|------|
| Phase A | 가변 카메라 capture/연동 — config 예제, YOLO batch 통합, target_fps 자동 | 부분 완료 |
| Phase B | UI 가변 그리드 — `_grid_layout()`, 풀뷰 토글, 화구 카드 카메라 배지 | 완료 |
| Phase C | 카메라 추가/제거 UX — 오프라인 감지, `handle_config_reloaded` 확장 | 완료 |
| Phase D | 운영 안정성 — 자동 재연결, 알림 사운드, 24시간 테스트 | 진행 중 |
| Phase E | 진단 도구 다카메라 확장 — `diag_perf.py` 신규 | 미착수 |
| Phase F | 운영자 친화 설정 UI — 슬라이더, 인라인 편집, 실시간 적용 | 미착수 |

---

## 12. 실패 기록 — Phase 3 주파수 분석 & MaskIoU

재도입 방지를 위해 실패 이유를 상세히 기록한다.

### Phase 3 — 주파수 분석 실패 (2026-04-05 확정)

**시도**: 딸랑이 진동 주파수(1~8Hz)를 IIR bandpass 필터로 추출하려 했으나 4가지 신호 모두 실패.

| 시도 | 신호 | 실패 원인 |
|------|------|---------|
| 픽셀 밝기 FFT | bbox 평균 밝기 | 주방 조명이 1~8Hz로 변동 → 오탐 |
| YOLO 중심점 FFT | 중심점 x좌표 | YOLO 정수 좌표 → 0.5px 진폭이 양자화 노이즈에 묻힘 |
| 전체 mean_flow_x FFT | bbox 평균 수평 flow | 배경 픽셀이 방향 신호 희석 → 평균 ≈ 0 |
| 마스크 masked_flow_x | 마스크 픽셀 평균 수평 flow | 마스크 추출률 10~20%, amp 평균 0.054 (임계값 0.3 미달) |

**결론**: Phase 2만으로 충분 (Burner 9 진동 norm_rms 30.7% vs Burner 10 정지 2.7%, 약 10배 차이).

`core/frequency_filter.py` 파일은 보존되어 있으나 config에서 `"enabled": false`로 비활성화.

### MaskIoU 폐기 이유 (2026-04-03 확정)

MaskIoU 방식(연속 프레임 간 segmentation 마스크의 IoU 변화량으로 진동 측정)의 근본 한계:

- YOLO seg 마스크가 프레임당 10~20%만 추출됨 (연기·조명에 따라 누락 심각)
- IoU 신호 자체가 너무 희박해 window 기반 판정이 불안정
- NCC 대비 장점(3D 회전 감지, 연기 FP 제거)이 있었으나 신호 희박 문제가 더 컸음

**HybridVibrationTracker → NCC(FrameDiffTracker) → MaskIoU → OpticalFlow**로 진동 감지 방식이 변화했으며, 최종적으로 OpticalFlow + mask 내부 RMS가 가장 안정적임을 확인.

---

## 13. 설계 변경 전체 이력

> "왜 바꿨나"를 기록. git에 없는 의사결정 맥락.

| 날짜 | 대상 | 변경 내용 & 이유 |
|------|------|----------------|
| 2026-03-06 | state_machine.py | 4상태 → 6상태 (STEAMING_FIRST/SECOND, DONE_FIRST/SECOND) — 초벌/재벌 2단계 사이클 반영 |
| 2026-03-06 | calibration.py | 화구 수 고정 → 드래그 기반 동적 지정 — 매장마다 화구 수/위치가 다름 |
| 2026-03-07 | frame_processor.py | read_frames() / detect_and_update() 분리, 배치 YOLO 추론 — 화구 수 무관 1회 추론으로 성능 확보 |
| 2026-03-09 | frame_processor.py | HybridVibrationTracker 도입: EMA 보정 + 템플릿 매칭 기반 진동 판별 |
| 2026-03-11 | main.py, config | Mac/Windows 호환성 버그 수정 (`apply_source_overrides` 복구, 웹캠 기본화) |
| ~2026-03-26 | frame_processor.py | **HybridVibrationTracker → NCC(FrameDiffTracker) 전환** — 정지 딸랑이 오탐(FP) 문제 |
| 2026-03-31 | dataset/ | Roboflow Segmentation 재라벨링 (129장) + augment_dataset.py 증강 (train 3,090장) |
| 2026-03-31 | train.py | yolov8n-seg.pt 학습으로 전환 — mask_xy 기반 감지를 위해 segmentation 모델 필요 |
| 2026-03-31 | frame_processor.py | **NCC(FrameDiffTracker) → MaskIoU 전환** — 3D 원형 회전 감지, 연기·조명 FP 해결 시도 |
| 2026-04-02 | stabilizer.py | **Phase 1 Stabilizer 구현** — LK + RANSAC + EMA. 카메라 흔들림이 optical flow 오탐 유발 |
| 2026-04-02 | optical_flow.py | **Phase 2 OpticalFlowDetector 구현** — Farneback dense flow + EMA + window 투표 |
| 2026-04-03 | frame_processor.py | **MaskIoU 완전 제거**, Phase 1+2 통합 — MaskIoU 근본 한계: 마스크 추출률 10~20%로 신호 희박 |
| 2026-04-03 | frequency_filter.py | Phase 3 FrequencyAnalyzer 구현 — IIR bandpass(1~8Hz) + EMA amplitude |
| 2026-04-05 | — | **Phase 3 실패 확정** — 4가지 신호 모두 실패 (상세 내용 §12 참조) |
| 2026-04-12 | augment_dataset.py | 증강 파이프라인 개선 — RandomRotate90 → Rotate(±40°, p=0.8), valid 원본 복사만으로 변경 |
| 2026-04-12 | train.py | close_mosaic 20→30, workers=4 추가 (발열 완화) |
| 2026-04-12 | models/pot_seg.pt | **재학습 완료** — 87 epoch (best: 71), mAP50(B) 75.4%→85.0%, mAP50(M) 70.8%→81.0% |
| 2026-04-19 | core/optical_flow.py | **bbox → mask_xy 기반 RMS 계산으로 전환** — crop 위치는 bbox center EMA 유지, RMS만 mask 내부 픽셀로. mask centroid EMA는 mask 없는 프레임마다 bbox center로 끌려 oscillation → FP 폭증, 즉시 폐기 |
| 2026-04-19 | main.py, sources/ | **영상 파일 재생 fps 동기화** — `round(video_fps / target_fps)` 프레임 스킵. 파일 소스에서도 실제 카메라와 동일한 시간축 유지 |
| 2026-04-19 | core/optical_flow.py | **RMS 정규화 도입 초안** — `normalized_rms = rms / sqrt(bbox_w × bbox_h)`. 카메라 가까운 딸랑이 FP 문제 → 2026-05-05에 방식 변경 |
| 2026-05-05 | ui/ui_display.py | **UI 전면 재설계** — 브랜드 테마("길가옆에 누룽지 삼계탕"), 인앱 캘리브레이션(F키), cv2 창 제거, pygame 단일 창 통합 |
| 2026-05-05 | core/optical_flow.py | **RMS 정규화 방식 확정** — `norm_rms = raw_rms × ref_diag / bbox_diag`. 값 수렴(상쇄) 없고 threshold 불변, 해상도·줌 독립 |
| 2026-05-05 | config/store_config.json | **rms_threshold 0.20 확정** — 500프레임 정지 영상 분석, noise p90=0.19. 이전 0.6은 실질적으로 감지 불가였음 |
| 2026-06-04 | ui/ui_display.py | **다카메라 UX 통합 대시보드 개편** — 균등 그리드 뷰, Cam-N 배지, ROI 클릭 선택+DEL, 토스트 메시지 |
| 2026-06-14 | state_machine.py 외 7파일 | **pot_absent_threshold 기본값 30 통일** — `__init__`=60 vs `add`=30 불일치. 실동작 60→30 변경 (반응성 ~1초) |
| 2026-06-14 | core/state_machine.py | **EMPTY 전환 시 `_reset_timer()` 호출** — DONE_FIRST/WAIT_SECOND/DONE_SECOND → EMPTY 시 내부 타이머 필드 stale 잔류 버그 수정 |
| 2026-06-14 | CLAUDE.md | **문서 7상태 동기화** — 코드는 이미 7상태인데 문서는 구버전 6상태. 상태표·전환 규칙·DONE_FIRST 냉각 단계 설명 갱신 |
| 2026-06-15 | core/state_machine.py | **`manual_start()` DONE_FIRST → '바로 재벌'** — 기존 `WAIT_SECOND` 점프에서 즉시 `_start_second()`(재벌 카운트다운 시작)로 변경 |
| 2026-06-15 | core/state_machine.py | **`seconds_since_done` / `is_counting` property 추가** — UI 유동 정렬(카운트다운 임박순 상단) 인터페이스 |
| 2026-06-15 | make_sounds.py, assets/sounds/ | **알림음 합성·음량 강화** — 초벌·재벌 공통 2시점(`warn_30s`, `complete`) WAV. 주방 소음 대응: 2·3배음 혼합, 음량 0.9 |

---

## 14. 개발 Phase별 완료 이력

### Phase 0 — 데이터 준비 및 라벨링 ✅

- Roboflow 라벨링 v3, 152장 증강 / 138 train + 14 val
- 1차 부트스트랩 학습 → mAP@0.5=95.3% (2026-03-07, detection 모델)

### Phase 1 — 코드 구현 ✅

- 6상태 상태머신 + 타이머 잠금
- ROI 캘리브레이션 (드래그 기반)
- 배치 YOLO 추론 (15fps)
- 딸랑이 독점 할당 (x축 거리 기반 greedy)
- body_ttl: 밥솥 미감지 시 15프레임 메모리 유지
- Segmentation 재라벨링 및 yolov8n-seg 학습 (2026-03-31)

### Phase 2 — 진동 감지 성능 튜닝 ✅

- MaskIoU → OpticalFlow 전환 (2026-04-02~05)
- Phase 1 Stabilizer 통합
- EMA 스무딩 튜닝 (오탐 억제)
- Phase 3 주파수 분석 시도 및 실패 결론 (→ 비활성화)
- 모델 재학습 완료 (2026-04-12, mAP50 85.0%)

### 다카메라 UX ✅ (2026-06-04)

- 균등 그리드 뷰, Cam-N 배지, ROI 클릭/DEL/ID 재정렬
- 운영자용 `run.bat` 더블클릭 실행
- 헤드리스 + 스크린샷 플래그
- ROI 합집합 면적 진단 로그
- 동적 margin 자동 산정

### 백엔드 정합성 정리 ✅ (2026-06-14~15)

- pot_absent_threshold 통일 (60→30)
- EMPTY 전환 시 `_reset_timer()` 호출
- `manual_start()` '바로 재벌' 지원
- `is_counting` / `seconds_since_done` property 추가
- 알림음 3종 합성 (`warn_30s`, `complete`, `cam_lost`)

### 현재 위치

```
Phase 0 ✅ → Phase 1 ✅ → Phase 2 ✅ → 모델 재학습 ✅ → 다카메라 UX ✅ → 현장 테스트 ⏳
```

---

## 15. 매장 셋업 & 운영 가이드

### 초기 셋업 (1회)

1. **카메라 N대 USB 연결**
   - 노트북 좌/우/후면 / USB-C에 **분산**해서 꽂기 (대역폭 경쟁 방지)
   - 가능하면 외장 전원 USB 3.0 허브

2. **노트북 절전 모드 OFF**
   - 전원 옵션 → "절대 절전 안 함"

3. **소프트웨어 설치 (1회)**
   ```powershell
   winget install astral-sh.uv
   git clone <repo-url>; cd Pressure_Chicken
   uv sync
   ```

4. **`run.bat` 실행** → N 카메라 그리드 표시

5. **F키로 캘리브레이션**
   - 각 카메라 영역에 화구 위치를 마우스 드래그
   - 잘못 그렸으면 클릭 후 `Del`
   - `Enter` 저장 → 핫 리로드

6. **콘솔 진단 로그 확인**
   - 모든 카메라가 🟢 또는 🟡이면 OK
   - 🔴이면 카메라 추가 또는 화구 분담 재배치

7. **카메라 노출 수동 고정** (`config/store_config.json`의 `sources[].exposure`)
   - 자동 노출은 false positive의 주범 — 매장 광량에 맞춰 `-3` ~ `-7` 시도

8. **20~30초 워밍업 후 운영 시작**

### 매일 실행

**바탕화면에 `run.bat` 바로가기를 더블클릭하면 끝.**

`run.bat` 동작 순서:
1. `uv` 설치 확인
2. `.venv` 없으면 자동 `uv sync`
3. `main.py` 실행

### 자주 발생하는 문제

| 증상 | 원인/대처 |
|------|-----------|
| `Error: -1072875772` (Windows) | MSMF 백엔드 오류 — 이미 DSHOW 강제로 자동 처리됨 |
| 카메라가 안 잡힘 | 다른 앱(Teams, Zoom)이 점유 중인지 확인 |
| 화면이 검고 ⛔ 신호 없음 | USB 다시 꽂기 → 5초 대기 → 안 되면 프로그램 재시작 |
| 정지 상태인데 진동 감지됨 | 자동 노출이 원인 가능 — `exposure` 값 수동 고정 |
| ROI 합집합 65% 초과 경고 | 화구를 다른 카메라로 분담하거나 카메라 추가 |
| AI 모델 없음 경고 | `models/pot_seg.pt` 가 있는지 확인 |

### Windows 웹캠 MSMF 에러 해결 원칙

`sources/video_source.py`에서 `cv2.VideoCapture(index, cv2.CAP_DSHOW)` 로 DirectShow 강제 지정. config의 `type: "camera"` 사용 시 자동 적용.

---

## 16. 프로젝트 파일 구조

```
Pressure_Chicken/
├── README.md                    # 빠른 시작 가이드
├── CLAUDE.md                    # 시스템 설계 & 불변 지식
├── ACTION_PLAN.md               # 진행 이력 & 미결 과제
├── MULTI_CAMERA_PLAN.md         # 다카메라 확장 설계
├── 0612UI작업.md                 # UI 개선 작업 명세서
├── PROJECT_OVERVIEW.md          # 이 파일 — 웹 공유용 종합 문서
│
├── main.py                      # 진입점
├── run.bat                      # 운영자용 더블클릭 실행 스크립트
├── train.py                     # YOLO 학습 스크립트
├── extract_frames.py            # 영상에서 프레임 추출
├── augment_dataset.py           # 데이터 증강 파이프라인
├── diag_rms.py                  # RMS 진단 스크립트 (FP 원인 분석)
├── make_sounds.py               # 알림음 WAV 합성 스크립트
│
├── core/
│   ├── state_machine.py         # 7상태 FSM + 타이머 + 잠금 로직
│   ├── frame_processor.py       # Phase 1+2 통합, body/weight 매칭
│   ├── detector.py              # YOLO-seg 추론 래퍼 (mask_xy 포함)
│   ├── stabilizer.py            # Phase 1: LK+RANSAC+EMA 흔들림 보정
│   ├── optical_flow.py          # Phase 2: Farneback RMS + EMA + window
│   └── frequency_filter.py      # Phase 3: IIR bandpass (현재 비활성화)
│
├── sources/
│   ├── video_source.py          # 카메라/파일 입력 추상화
│   └── camera_utils.py          # 카메라 전환 유틸
│
├── ui/
│   └── ui_display.py            # pygame 통합 대시보드
│
├── assets/
│   └── sounds/
│       ├── warn_30s.wav         # 완료 30초 전 예고음
│       ├── complete.wav         # 완료음
│       └── cam_lost.wav         # 카메라 연결 끊김 경고음
│
├── tests/
│   └── compare_phase3.py        # Phase 3 시각 비교 뷰어
│
├── models/
│   └── pot_seg.pt               # 학습된 segmentation 모델 (6.5MB)
│
├── dataset/
│   └── dataset.yaml             # YOLO 데이터셋 정의
│
├── config/
│   ├── store_config.json        # 현재 운영 설정
│   └── examples/                # 카메라 수별 예시 config
│       ├── store_1cam.json
│       ├── store_3cam.json
│       ├── store_4cam.json
│       ├── store_5cam.json
│       ├── store_4cam_video.json
│       └── store_4cam_video_with_burners.json
│
└── docs/
    ├── MULTICAM_UX_WORK_LOG.md  # 다카메라 UX 작업 로그
    ├── pipeline_diagram.py       # 파이프라인 다이어그램 생성
    ├── pipeline.png
    ├── phase1.png
    └── phase2.png
```

---

## 17. 의존성 & 실행 방법

### pyproject.toml

```toml
[project]
name = "pressure-timer"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "ultralytics>=8.0.0",    # YOLOv8 (YOLO 추론)
    "opencv-python>=4.8.0",  # 영상 처리, optical flow, stabilizer
    "pygame>=2.5.0",         # UI 렌더링, 알림음 재생
    "numpy>=1.24.0",         # 배열 연산
]
```

패키지 매니저: `uv` (pip보다 빠르고 .venv 자동 관리)

### 실행 명령어

```powershell
# 기본 실행 (config/store_config.json 자동 로드)
uv run python main.py

# 다른 config 지정
uv run python main.py --config config/examples/store_4cam.json

# 영상 파일로 시뮬레이션 (실제 카메라 없이)
uv run python main.py --source-0 raw/Side_01.mov

# 다중 영상 시뮬 — 4카메라 환경 재현
uv run python main.py --config config/examples/store_4cam_video.json

# 헤드리스 + 스크린샷 (UI 검증용, GUI 없이)
uv run python main.py --config <cfg> --test 30 --headless --screenshot out.png

# RMS 진단 (정지 노이즈 확인 / threshold 튜닝)
uv run python diag_rms.py --frames 300
uv run python diag_rms.py --burner 3 --frames 150 --skip 30

# 알림음 재합성
uv run python make_sounds.py

# 학습 데이터 증강
uv run python augment_dataset.py

# 모델 학습
uv run python train.py
```

### 인터넷 연결 요건

- **학습 시**: `yolov8n-seg.pt` 베이스 모델 최초 1회 다운로드 필요
- **운영 시**: 완전 로컬. 인터넷 불필요.

---

## 18. 현재 성능 지표 & 파라미터

### YOLO 모델 성능 (2026-04-12, 87 epoch)

| 지표 | 값 |
|------|-----|
| mAP50(B) Bounding Box | **85.0%** |
| mAP50(M) Mask | **81.0%** |
| mAP50-95(B) | 67.8% |
| mAP50-95(M) | 54.4% |
| 모델 크기 | 6.5MB |
| 파라미터 수 | 3.26M |

### 현재 optical_flow 파라미터 (store_config.json)

```json
"optical_flow": {
  "rms_threshold": 0.20,
  "rms_ema_alpha": 0.35,
  "window_frames": 25,
  "trigger_frames": 14,
  "normalize_rms": true,
  "normalize_ref_diag": 40.0
}
```

**threshold 선정 근거**:
- 정지 딸랑이 noise floor: norm_rms ≈ 0.015~0.13 (p99 ≈ 0.32)
- threshold 0.20 = noise p90(0.19) 바로 위
- window 투표(14/25)로 FP 추가 차단
- `normalize_ref_diag=40.0`: 현재 영상 기준 평균 딸랑이 bbox 대각선(~37px). 해상도·줌 변경 시 재측정 필요.

### 백엔드 검증 결과 (2026-06-15, ALL PASS)

| 검증 항목 | 결과 |
|----------|------|
| '바로 재벌': DONE_FIRST에서 `manual_start()` → 즉시 `POT_STEAMING_SECOND`, 재벌 카운트다운 시작 | ✅ |
| `is_counting`: STEAMING 중 True / 정지 상태 False | ✅ |
| `seconds_since_done`: STEAMING 중 -1 / 완료 후 ≥0 | ✅ |
| `pot_absent_threshold` 기본값 30 | ✅ |
| `DONE_SECOND` 이탈(연속 미감지) → EMPTY + `_reset_timer`로 타이머 필드 정리 | ✅ |
| debounce: threshold 미만 순간 가림(10<30)은 상태 유지 | ✅ |

---

## 19. 미결 과제 & 로드맵

### 최우선: 현장 카메라 라이브 테스트

- [ ] 딸랑이 움직임 → STEAMING 자동 전환 안정 확인
- [ ] 초벌 완료 → DONE_FIRST → 재벌 딸랑이 재감지 → 재벌 시작 전체 사이클 확인
- [ ] 타이머 잠금 확인 (사람 가림, 연기 발생 시 타이머 유지)
- [ ] 밥솥 이탈 후 재거치 → EMPTY → POT_IDLE 전환 확인

### 테스트 시나리오

| 시나리오 | 기대 결과 | 검증 항목 |
|---------|---------|-------|
| 정지 딸랑이 3분 관찰 | STEAMING 전환 없음 | FP 검증 |
| 딸랑이 진동 시작 후 10초 이내 | STEAMING 전환 | FN 검증 |
| 딸랑이 미세 진동 (막 시작할 때) | STEAMING 전환 | FN 검증 (어려운 케이스) |
| 손으로 밥솥 5초 가림 후 제거 | STEAMING 유지 | 잠금 로직 |
| 수증기 자욱한 구간 | STEAMING 유지 | 잠금 로직 |
| 밥솥 이탈 후 재거치 | EMPTY → POT_IDLE | 상태 전환 |

### UI 개선 (0612UI작업.md 기준)

- [ ] 작업 A: dev 모드에서 화구 번호 크게
- [ ] 작업 B: '재벌대기' 표기 중복 제거
- [ ] 작업 C: RMS 수치는 dev 모드에서만
- [ ] 작업 D: DONE_FIRST 버튼 라벨 "건너뜀" → "바로 재벌"
- [ ] 작업 E: 알림음 재생 연결 (warn_30s / complete / cam_lost)
- [ ] 작업 F: 유동 정렬 (카운트다운 임박순 좌상단)
- [ ] 작업 G: 분할선 드래그 / 영상 숨김 토글 / 카드 크기 패널폭 연동
- [ ] 작업 H: 카메라 연결 끊김 경고음

### 다카메라 Phase 남은 항목

- [ ] Phase D: 자동 재연결, 장치명 기반 식별, 24시간 연속 운용 테스트
- [ ] Phase E: `diag_perf.py` 신규 — 카메라 N대 환경에서 처리 지연·CPU 사용률 측정
- [ ] Phase F: 운영자 친화 설정 UI — 슬라이더 실시간 적용, 화구 카드 인라인 편집

---

*최종 업데이트: 2026-06-23*
*상태: 백엔드 완료 → UI 작업 + 현장 1차 테스트 대기 중*
