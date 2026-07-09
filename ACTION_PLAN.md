# 압력밥솥 타이머 — 액션플랜

> 최종 업데이트: 2026-07-09 (딸랑이 오분류 "와리가리" 해결 + 카메라 해상도 고정) | 현재 단계: 매장 1차 테스트

---

## 2026-07-09 작업 — 딸랑이/vent 오분류로 인한 상태 와리가리 해결

### A. weight/vent 통합 매칭 + ROI 클리핑 (`core/frame_processor.py`)
- **문제**: YOLO가 vent(class=2)를 weight(class=1)로, 또는 그 반대로 오분류하는 경우가 있어
  프레임마다 어느 쪽이 채택되는지 흔들리면 STEAMING 판정이 오락가락(와리가리)함. 실측으로
  확인된 별도 사례로, body 검출이 비정상적으로 크거나 어긋나면 화구 window(±15%)가 자기
  ROI를 벗어나 옆 화구 영역까지 침범 — 13번 화구가 11번 화구의 vent를 자기 weight로
  채가는 화구 간 오염도 발견됨.
- **조치**:
  1. weight-class와 vent-class 검출을 하나의 후보 풀로 합쳐, 화구당 **면적이 가장 큰 것
     하나만** 딸랑이로 채택 (실측 라벨 면적 중앙값 weight(0.0007) > vent(0.00034), 약 2배
     — "더 큰 쪽이 진짜 딸랑이"라는 도메인 사전지식으로 모델 class 출력을 재검증). 후보가
     단독일 때는 기존과 동일하게 그대로 선택되어 정상 케이스엔 영향 없음.
  2. 화구 window(±15% margin)를 캘리브레이션된 자기 ROI와 교집합으로 클리핑 후 매칭 —
     화구 간 후보 오염(위 13/11번 사례) 방지.
  3. dev 모드 오버레이 + `data_logger.py` 로그에 vent 박스·weight 후보 박스·최종 채택
     박스를 모두 기록해, 향후 유사 오분류를 좌표 비교로 진단 가능하게 함
     (`BurnerStateMachine.vent_count`/`weight_class_count` 필드 추가).
- **변경 파일**: `core/frame_processor.py`, `core/state_machine.py`, `core/data_logger.py`,
  `ui/ui_display.py`(dev 모드 vent 박스 오버레이, 자홍색), `main.py`(logger.update에 processor 전달).

### B. Stabilizer 해상도 변경 대응 (`core/stabilizer.py`)
- **문제**: 카메라 재연결 등으로 프레임 해상도가 바뀌면 이전 프레임(`_prev_gray`)과 크기가
  달라 `cv2.calcOpticalFlowPyrLK`가 크기 불일치로 죽음.
- **조치**: 첫 프레임 조건에 `이전 프레임과 shape가 다른 경우`를 추가 — 특징점을 새로
  잡고 해당 프레임은 보정 없이 통과. `_smooth_dx`/`_smooth_dy`도 함께 리셋해 해상도 전환
  직후 스무딩 값이 이전 스케일 그대로 남는 것을 방지.
- **연동**: 위 2026-07-08 "카메라 해상도 고정 + 폴백" 항목으로 해상도가 폴백/재협상될 때
  이 로직이 없으면 크래시 위험이 있었음.

---

## 2026-07-08 작업 — 카메라 해상도 고정 + 폴백 (`sources/video_source.py`)

- **문제**: 카메라 해상도를 지정하지 않으면 드라이버 기본값(장치마다 제각각)으로 열려
  화구 ROI 캘리브레이션이 카메라 재부팅/교체마다 어긋남. 일부 카메라는 미지원 해상도
  요청 시 `isOpened()`는 `True`를 반환하면서 `read()`가 계속 실패 — `cap.get()`으로
  협상 결과만 확인하는 것으로는 불충분함이 확인됨.
- **조치**: `_open_camera_with_fallback()` 신설 — 기본 `1920×1080`(소스별 `frame_width`/
  `frame_height` config로 override 가능)으로 열기 시도 후 **실제 프레임 한 장을 읽어보는
  것까지 성공**해야 채택. 실패 시 `1280×720` → `640×480` 순차 폴백, 실제 열린 해상도를
  콘솔 로그로 출력.
- **부가**: Windows MSMF 오픈 로직(`CAP_PROP_HW_ACCELERATION=NONE`, 밝기 블리칭 버그
  방지용)과 통합해 `_open_camera_cv()`로 정리.
- **변경 파일**: `sources/video_source.py`.

---

## 2026-06-16 작업 — UI 통합 캔버스 재설계 + 타이머값 통일 + gamma 정리

### A. 화구 상태머신 6→7상태 (WAIT_SECOND 분리)
- **문제**: 기존 `DONE_FIRST`가 "초벌 완료 냉각"과 "재벌 대기(진동 재활성)"를 `_seen_rest_after_first`
  플래그로 한 상태 안에서 같이 처리 → 딸랑이가 잠깐만 멈춰도 곧바로 재벌로 오인 전환될 위험.
- **변경**: `DONE_FIRST`(냉각, 진동 완전 무시, `done_first_timeout` 후 자동 전환)와
  `WAIT_SECOND`(재벌 대기, 진동 재활성)를 별도 상태로 분리. `done_first_timeout` 기본 600초→
  이후 사용자 협의로 **120초(2분)** 로 조정.
- **부가**: `DONE_FIRST`/`WAIT_SECOND`/`DONE_SECOND`에서 pot 미감지 시 즉시 EMPTY로 가지 않고
  `pot_absent_threshold`(반칸 debounce, 기본 30프레임≈1초 + `_body_ttl` 15프레임 별도 더해
  총 약 5초 여유) 만큼 연속 미감지일 때만 EMPTY 확정 — 손/김에 잠깐 가려도 타이머 안 날아감.
- **변경 파일**: `core/state_machine.py`(상태 enum·전환로직·`_pot_absent_count`),
  `core/frame_processor.py`(WAIT_SECOND 진입 시 oflow 리셋), `main.py`, `ui/ui_display.py`,
  `config/store_config.json`.

### B. UI 전면 재설계 — 좌/우 분리 패널 폐지 → 통합 캔버스
- **배경**: 매장 실제 배치(ㄱ자, 화구 두 블록 분리 등)를 화면에 그대로 재현하고 싶다는 요구.
  기존 구조(좌측 카메라 그리드 / 우측 화구 카드 패널, 카드는 "남은 시간 적은 화구가 위로
  뜨는" 자동 정렬 고정 4열)로는 표현 불가능.
- **변경**: 카메라 박스와 화구 카드를 **하나의 캔버스에 반칸(half-cell) 단위 그리드로 함께
  자유 배치**하는 구조로 전면 교체.
  - `layout_cols`(반칸 열 수, 기본 16) 기준으로 `_half_cell_w/h` 매 프레임 계산
  - 화구 카드 = 2×2 반칸 고정, 카메라 박스 = `layout_size`로 칸 수 자유 지정
  - **배치 모드(`L`키)**: 카드/카메라 드래그 이동(좌측상단 그랩 오프셋 보존, 드래그 중
    미리보기가 항상 스냅된 실제 위치에 그려짐 — "보이는 대로 놓인다"), 카메라 우하단
    ↘ 핸들로 반칸 단위 리사이즈. `ENTER` 저장(`grid_pos`/`layout_pos`/`layout_size`) / `ESC` 취소
  - 카드 자동 정렬(urgency sort) 완전 폐지 → 저장된 위치 고정 표시
  - 캘리브레이션(F)·카메라 풀뷰(더블클릭)·카메라 추가/제거 등 기존 기능은 새 캔버스 좌표계에
    맞춰 재구현 (동작 자체는 동일)
- **화구 카드 재설계**: 화구 번호를 카드 내 **가장 크고 눈에 잘 보이는 요소**로 변경
  (카드 높이에 비례하는 동적 폰트 크기, 상단 대부분 차지, 가로 중앙 정렬). 상태/타이머는
  번호 아래 보조 정보 한 줄로 통합. 폰트는 크기별 캐싱(`_dynamic_font`)으로 성능 영향 없음.
- **변경 파일**: `ui/ui_display.py`(대규모 재작성), `config/store_config.json`
  (`layout_cols`, 화구 `grid_pos` half-cell 단위로 재산정, 소스 `layout_pos`/`layout_size` 추가).
- **검증**: 헤드리스(`SDL_VIDEODRIVER=dummy`) 환경에서 일반/배치/캘리브레이션/풀뷰 렌더링,
  카드·카메라 드래그, 카메라 리사이즈, 저장까지 단위 테스트로 확인. 실제 드래그 체감은
  디스플레이 있는 환경에서 추가 확인 필요.

### C. optical_flow `normalize_gamma` 실험 도입 → 데이터 부재로 1.0 환원
- 거리별(딸랑이 bbox 크기별) 보정 강도를 조절하는 `normalize_gamma` 파라미터가 코드에
  추가되어 있었음(0.6) — 단, 화구별 실제 bbox_diag 분포를 측정한 근거 없이 정해진 값.
- gamma<1.0은 카메라가 가까운 화구(bbox 큼)는 더 민감하게, 먼 화구(bbox 작음)는 더
  둔감하게 만드는 식으로 화구별로 다르게 작용 → 근거 없이 두면 화구마다 감도가
  들쭉날쭉해질 위험.
- **결정**: 데이터로 확인하기 전까지 `normalize_gamma: 1.0`(완전 보정, 기존 검증된 방식)으로
  환원. `gamma` 적용 코드 자체는 유지 — 향후 `diag_rms.py --frames 300`으로 화구별 `bbox_d`
  분포를 실측해 ref_diag(40px)와 크게 벗어나면 데이터 기반으로 재조정.
- **변경 파일**: `config/store_config.json` (`normalize_gamma` 0.6→1.0).

### D. 타이머 값 전체 화구 통일
- 인앱 캘리브레이션(F키 드래그)으로 신규 추가된 화구(5~15번)가 코드에 박힌 구버전 기본값
  (`countdown_second=300`, `done_first_timeout=600`)을 그대로 가져가 1~4번(270/120)과 값이
  갈라져 있었음.
- **조치**: 전체 화구를 `countdown_second=270`(4분30초), `done_first_timeout=120`(2분)으로
  통일. 캘리브레이션 신규 화구 기본값(`ui/ui_display.py`)과 `main.py`의 config 폴백 기본값도
  동일하게 맞춰서, 앞으로 추가되는 화구도 자동으로 통일되도록 함.

---

## 2026-06-15 백엔드 검증 결과 (ALL PASS)

상태머신 변경(바로재벌 · 신규 property · pot_absent 30 통일 · EMPTY 정리)을 단위 검증. **12개 항목 전부 PASS.**

| 검증 항목 | 결과 |
|----------|------|
| '바로 재벌': DONE_FIRST에서 `manual_start()` → 즉시 `POT_STEAMING_SECOND`, 재벌 카운트다운 시작 | ✅ |
| `is_counting`: STEAMING 중 True / 정지 상태 False | ✅ |
| `seconds_since_done`: STEAMING 중 −1 / 완료 후 ≥0 | ✅ |
| `pot_absent_threshold` 기본값 30 | ✅ |
| `DONE_SECOND` 이탈(연속 미감지) → EMPTY + `_reset_timer`로 타이머 필드 정리 | ✅ |
| debounce: threshold 미만 순간 가림(10<30)은 상태 유지 | ✅ |

검증 방식: `core.state_machine` 단위 호출(countdown=0으로 즉시 완료 시뮬). 1회성 검증이라 임시 스크립트는 보관하지 않음.
알림음은 청취 검증 필요(주방 소음 환경에서 들리는지) — UI 재생 연결 후 현장에서 확인.

---

## 2026-06-14 작업 — 상태머신 정합성 정리 & 문서 동기화

내부 로직 점검 중 발견한 정합성 문제 3건을 정리. 감지 알고리즘(Phase 1/2)은 무수정.

### A. `pot_absent_threshold` 기본값 불일치 → 30으로 통일
- **발견**: 기본값이 두 곳에서 어긋나 있었음 —
  `BurnerStateMachine.__init__`=60 vs `BurnerRegistry.add`=30.
- **실태**: `main.py`가 config 값(당시 전부 60)을 항상 명시 전달하므로 `add`의 `30`은
  **한 번도 실행되지 않는 죽은 기본값**이었음. 코드 전체에서 30이 박힌 곳은 `add` 단 한 곳,
  나머지(실동작·config 16화구·예제·docs)는 전부 60.
- **결정**: **30으로 통일** (반응성 ~30fps 기준 약 1초; 기존 60은 약 2초).
  → 완료 계열 상태에서 밥솥 이탈 후 EMPTY 복귀가 빨라짐 (실동작 변경 사항).
- **변경 파일**: `state_machine.py`(`__init__` 60→30), `main.py`(초기화·핫리로드 2곳),
  `ui/ui_display.py`(캘리브 신규 화구), `tests/sim_camera_select.py`,
  `config/store_config.json`(16화구), `config/examples/store_4cam_video_with_burners.json`(8화구),
  `docs/pipeline_diagram.py`.

### B. EMPTY 전환 시 타이머 상태 미정리 → `_reset_timer()` 호출로 수정
- **문제**: DONE_FIRST / WAIT_SECOND / DONE_SECOND → EMPTY 전환 시 `_pot_absent_count`만
  0으로 리셋하고 `_done_first_end` 등 내부 타이머 필드는 stale 값으로 남았음.
- **조치**: 세 전환 지점 모두 `_reset_timer()` 호출로 변경 (count·countdown_end·done_time·done_first_end 일괄 정리).
  현재는 화면 표시 버그로 이어지진 않았으나(EMPTY에서 해당 필드 미참조) 잠재 위험 제거.

### C. 문서 7상태 동기화 (`CLAUDE.md`)
- 코드는 이미 **7상태**(WAIT_SECOND 포함)인데 CLAUDE.md는 구버전 6상태 표를 유지하고 있었음.
- 상태 정의 표를 7상태로 갱신 + 실제 색상 RGB·진동 감지 활성 여부 명기.
- 타이머 사이클에 **DONE_FIRST 냉각 단계**(완료 직후 잔여 진동 무시 → done_first_timeout 후 WAIT_SECOND) 설명 추가.
- 상태 전환 규칙 표·`pot_absent_threshold` 설명 추가.
- `store_config.json` 예시 보강: `done_first_timeout`, `pot_absent_threshold`(30), `normalize_rms`/`normalize_ref_diag` 추가, `rms_threshold` 0.5→0.20 동기화.

---

## 2026-06-04 작업 — 다카메라 UX + 매장 배포 준비

매장 노트북(4 카메라) 이관 직전 완료. 감지 알고리즘(Phase 1/2/State Machine)은 무수정.

### A. UI/UX 통합 대시보드 개편 (`ui/ui_display.py`, `main.py`)
- 다카메라 균등 그리드 뷰 (1대=풀화면, 2대=1×2, 3~4대=2×2, 5~6대=3×2 …)
- 각 셀 헤더: `[Cam-N]` + 연결 상태 LED (● 연결됨 / ⛔ 프레임 없음)
- 캘리브 드래그 → 클릭한 셀의 `source_id` 자동 귀속
- 캘리브 ROI 클릭 선택 + DEL 삭제 + ID 자동 재정렬
- 화구 카드에 `Cam-N` 뱃지 + 진동 게이지 항상 표시
- 우측 패널 `[+ 추가]` / `[− 제거]` 버튼 (가용 카메라 자동 탐색)
- 토스트 메시지 (카메라 추가/제거 등 즉시 피드백)

### B. 매장 배포 준비
- `.gitignore` 보안 보강 — `raw/*.xlsx`, `raw/~$*`, `raw/*.csv` (거래처 정보 차단)
- **ROI 합집합 면적 진단 로그** — 콘솔에 카메라별 화구 분담 부담 출력 ([frame_processor.py](core/frame_processor.py))
  ```
  [diag] Cam-0: burners=2, ROI 합집합=34.7%  → ✓ 가벼움
  [diag] Cam-1: burners=4, ROI 합집합=78.3%  → ⚠⚠ 카메라 추가 검토
  ```
- **동적 margin** — ROI 합집합 crop margin을 `max(30, min(50, 짧은변 × 10%))` 로 자동 산정. 작은 ROI 환경에서 YOLO 입력 면적 절감
- `run.bat` 운영자용 실행 스크립트 — uv 자동 체크 + .venv 첫 실행 시 sync + 더블클릭 실행
- 4영상 시뮬 config 2종: `store_4cam_video.json` (빈 화구), `store_4cam_video_with_burners.json` (사전 화구 8개)
- `--headless` + `--screenshot` 플래그로 GUI 없이 UI 검증 가능
- 회귀 테스트 3 시나리오 모두 통과 (1cam·4cam+8burner·4cam+0burner)

상세 단계별 결과: [docs/MULTICAM_UX_WORK_LOG.md](docs/MULTICAM_UX_WORK_LOG.md)

---

## 현재 진행 위치

```
Phase 0 ✅  →  Phase 1 ✅  →  Phase 2 ✅  →  모델 재학습 ✅  →  다카메라 UX ✅  →  현장 테스트 ⏳
```

**확정된 감지 스택:**
- Phase 1: Stabilizer (LK + RANSAC + EMA warpAffine)
- Phase 2: OpticalFlowDetector (Farneback RMS + EMA + window 투표)
- Phase 3: 비활성화 (주파수 분석 — 신호 품질 부족으로 포기)

**현재 모델 성능 (2026-04-12 기준):**

| 지표 | 이전 | 현재 |
|------|------|------|
| mAP50(B) | 75.4% | **85.0%** |
| mAP50(M) | 70.8% | **81.0%** |
| mAP50-95(B) | — | 67.8% |
| mAP50-95(M) | — | 54.4% |

**현재 파라미터 (store_config.json):**
```json
"optical_flow": {
  "rms_threshold": 0.20,
  "rms_ema_alpha": 0.35,
  "window_frames": 25,
  "trigger_frames": 14,
  "normalize_rms": true,
  "normalize_ref_diag": 40.0,
  "normalize_gamma": 1.0
}
```
`rms_threshold` 단위: `norm_rms = raw_rms × (ref_diag / bbox_diag) ^ gamma` 기준 비율.
- 정지 딸랑이 noise floor: norm_rms ≈ 0.015~0.13 (p99 ≈ 0.32)
- threshold 0.20 = noise p90(0.19) 바로 위, window 투표(14/25)로 FP 추가 차단
- `normalize_ref_diag=40.0`: 현재 영상 기준 평균 딸랑이 bbox 대각선(~37px) 기준. 해상도·줌 변경 시 이 값 재측정.
- `normalize_gamma=1.0`(완전 보정, 기존 검증된 방식). 0.6 등으로 낮추면 보정 강도가 약해져
  화구별 카메라 거리에 따라 감도가 달라짐 — 데이터(실측 bbox_d 분포) 없이 변경하지 말 것.

---

## 미결 과제

### 최우선: 현장 카메라 라이브 테스트
- [ ] 딸랑이 움직임 → STEAMING 자동 전환 안정 확인
- [ ] 초벌 완료 → DONE_FIRST(냉각) → WAIT_SECOND(재벌대기) → 딸랑이 재감지 → 재벌 시작 사이클 확인
- [ ] 타이머 잠금 확인 (사람 가림, 연기 발생 시 타이머 유지)
- [ ] 밥솥 이탈 후 재거치 → EMPTY → POT_IDLE 전환 확인

### UI 통합 캔버스 (2026-06-16 재설계) — 실제 디스플레이 환경 확인 필요
- [ ] 배치 모드(`L`)에서 카드/카메라 드래그 체감 — 헤드리스 환경에선 로직만 검증, 실제 마우스
      드래그 시 스냅 동작이 자연스러운지 확인
- [ ] 카메라 리사이즈 핸들이 화면 배율(고DPI 등)에서도 클릭하기 충분히 큰지 확인
- [ ] 화구 15개 이상으로 캔버스가 세로로 길어질 때 창 크기/배치 모드 사용성 확인
- [ ] 딸랑이 진동 정규화 `normalize_gamma` 데이터 기반 재조정 — 매장에서 `diag_rms.py --frames 300`
      돌려 화구별 `bbox_d` 분포를 ref_diag(40px)와 비교, 차이가 크면 gamma 재검토

### Phase 3 — 카메라 가변 N대 확장 (1~8+ 대)
- 상세 설계: **[MULTI_CAMERA_PLAN.md](MULTI_CAMERA_PLAN.md)** 참조
- [ ] Phase A: 가변 카메라 capture/연동 (YOLO batch 통합, resize/fps 자동 조정)
- [ ] Phase B: UI 가변 그리드 (1/2/4/6대 자동 레이아웃 + 풀뷰 토글)
- [ ] Phase C: 카메라 추가/제거 UX (오프라인 감지, sources 핫 리로드)
- [ ] Phase D: 운영 안정성 (자동 재연결, 장치명 식별, 알림 사운드, 24시간 테스트)
- [ ] Phase E: 진단 도구 다카메라 확장 (`diag_perf.py` 신규)

### 테스트 시나리오

| 시나리오 | 기대 결과 | 검증 항목 |
|---------|---------|-------|
| 정지 딸랑이 3분 관찰 | STEAMING 전환 없음 | FP 검증 |
| 딸랑이 진동 시작 후 10초 이내 | STEAMING 전환 | FN 검증 |
| 딸랑이 미세 진동 (막 시작할 때) | STEAMING 전환 | FN 검증 (어려운 케이스) |
| 손으로 밥솥 5초 가림 후 제거 | STEAMING 유지 | 잠금 로직 |
| 수증기 자욱한 구간 | STEAMING 유지 | 잠금 로직 |
| 밥솥 이탈 후 재거치 | EMPTY → POT_IDLE | 상태 전환 |

---

## Phase별 완료 이력

### Phase 0 — 데이터 준비 및 라벨링 ✅
- Roboflow 라벨링 v3, 152장 증강 / 138 train + 14 val
- 1차 부트스트랩 학습 → mAP@0.5=95.3% (2026-03-07, detection 모델)

### Phase 1 — 코드 구현 ✅
- 6상태 상태머신 + 타이머 잠금
- ROI 캘리브레이션 (드래그 기반)
- 배치 YOLO 추론 (15fps)
- 딸랑이 독점 할당 (x축 거리 기반 그리디)
- body_ttl: 밥솥 미감지 시 15프레임 메모리 유지
- Segmentation 재라벨링 및 yolov8n-seg 학습 (2026-03-31)

### Phase 2 — 진동 감지 성능 튜닝 ✅
- MaskIoU → OpticalFlow 전환 (2026-04-02~05)
- Phase 1 Stabilizer 통합
- EMA 스무딩 튜닝 (오탐 억제)
- Phase 3 주파수 분석 시도 및 실패 결론 (→ 비활성화)
- 모델 재학습 완료 (2026-04-12, mAP50 85.0%)

---

## 설계 변경 이력

> **이 섹션은 "왜 바꿨나"를 기록한다. git에 없는 의사결정 맥락.**

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
| 2026-03-31 | frame_processor.py | **NCC(FrameDiffTracker) → MaskIoU 전환** — 3D 원형 회전 감지, 연기·조명 FP 해결 |
| 2026-04-02 | stabilizer.py | **Phase 1 Stabilizer 구현** — LK + RANSAC + EMA. 카메라 흔들림이 optical flow 오탐 유발 |
| 2026-04-02 | optical_flow.py | **Phase 2 OpticalFlowDetector 구현** — Farneback dense flow + EMA + window 투표 |
| 2026-04-03 | frame_processor.py | **MaskIoU 완전 제거**, Phase 1+2 통합 — MaskIoU 근본 한계: 마스크 추출률 10~20%로 신호 희박 |
| 2026-04-03 | frequency_filter.py | Phase 3 FrequencyAnalyzer 구현 — IIR bandpass(1~8Hz) + EMA amplitude |
| 2026-04-05 | — | **Phase 3 실패 확정** — 4가지 신호 모두 실패 (상세 내용 아래 참조) |
| 2026-04-12 | augment_dataset.py | 증강 파이프라인 개선 — RandomRotate90 → Rotate(±40°, p=0.8), valid 원본 복사만으로 변경 |
| 2026-04-12 | train.py | close_mosaic 20→30, workers=4 추가 (발열 완화) |
| 2026-04-12 | models/pot_seg.pt | **재학습 완료** — 87 epoch (best: 71), mAP50(B) 75.4%→85.0%, mAP50(M) 70.8%→81.0% |
| 2026-04-19 | core/optical_flow.py | **bbox → mask_xy 기반으로 전환 (확정)** — ① crop 위치: bbox 중심 EMA (`pos_alpha=0.3`) — mask 유무와 무관하게 항상 bbox center 사용 (mask centroid EMA는 mask 없는 프레임마다 bbox center로 끌려 oscillation 발생 → FP 폭증 원인, 즉시 폐기), ② RMS 계산: mask 폴리곤 내부 픽셀만. mask 없는 프레임은 bbox 전체 RMS fallback. 진단 결과: 80프레임 3개 스파이크만 발생(window 14 미달 → STEAMING 전환 없음), FP 해결 확인. |
| 2026-04-19 | main.py, sources/video_source.py | **영상 파일 재생 fps 동기화 구현** — 파일 소스 사용 시 `round(video_fps / target_fps)` 만큼 프레임 스킵. 실시간 카메라와 동일한 시간축 유지 목적. window_frames, EMA 등 파라미터가 실제 환경과 동일하게 적용됨. |
| 2026-04-19 | core/optical_flow.py | **RMS 정규화 도입 (bbox 크기 기준) — 초안** — `normalized_rms = rms / sqrt(bbox_w × bbox_h)`. 카메라와 가까운 딸랑이가 같은 정지 상태에서도 절대 RMS가 높게 나와 FP 발생. → 2026-05-05에 방식 변경됨. |
| 2026-04-19 | config/store_config.json | **rms_threshold 스케일 변경** — 0.5(절대px) → 조정 중. → 2026-05-05에 0.20으로 확정. |
| 2026-05-05 | ui/ui_display.py | **UI 전면 재설계** — 브랜드 테마 도입("길가옆에 누룽지 삼계탕", 옐로우/블랙). ① 우측 패널(400px): 브랜드 헤더 + 화구 카드 목록 + 단축키 버튼. ② 인앱 캘리브레이션(F2): 드래그로 ROI 직접 그리기, ENTER 저장, Z 취소 → 별도 `calibration.py` 실행 불필요. ③ 카메라 영상 오버레이: YOLO bbox/mask/RMS 수치 같은 창에 표시. ④ 비디오 일시정지(Space), Mask 토글(M), 카메라 전환(C) 단일 UI 내 통합. |
| 2026-05-05 | main.py | **단일 UIDisplay 통합, cv2 창 제거** — 기존 `draw_preview()` cv2 창 + pygame 이중 구조 폐기. UIDisplay 단일 창에서 영상 오버레이까지 처리. `--calibrate` CLI 플래그 제거(→ F2 키로 대체). `--test` 플래그 제거. `numpy` 직접 사용 제거. |
| 2026-05-05 | core/optical_flow.py | **RMS 정규화 방식 확정** — `norm_rms = raw_rms × ref_diag / bbox_diag`. 이전 방식(`/sqrt(w×h)`) 대비: ① 스케일 유지 — ref_diag로 다시 곱해 값이 0으로 수렴(상쇄)하지 않음. ② threshold 불변 — bbox 크기가 바뀌어도 0.20 그대로 사용. ③ 해상도/줌 독립 — 해상도 2배 → bbox 2배 → scale 0.5 → 보정됨. `bbox_diag` 계산을 jump 감지 블록 밖으로 이동해 항상 사용 가능하게 함. |
| 2026-05-05 | diag_rms.py | **bbox_d, norm_rms 컬럼 추가** — 화구별 실제 bbox 대각선과 정규화 후 RMS를 나란히 표시. 판정 기준도 normalize 설정에 따라 deform_rms ↔ norm_rms 자동 전환. old_rms 컬럼 제거. |
| 2026-05-05 | config/store_config.json | **rms_threshold 0.20 확정, normalize 옵션 추가** — 500프레임 정지 영상 분석 결과: noise p90=0.19. threshold 0.20 채택. `normalize_rms: true`, `normalize_ref_diag: 40.0` 추가. 이전 threshold 0.6은 정지 noise 최댓값(0.59)과 거의 같아 실질적으로 감지 불가였음. |
| 2026-06-14 | state_machine.py 외 7파일 | **pot_absent_threshold 기본값 30 통일** — `BurnerRegistry.add`=30 vs `__init__`=60 불일치. add의 30은 main.py가 config값을 항상 전달해 미사용(죽은 기본값)이었음. 반응성(~1초)을 위해 30으로 통일, 실동작 60→30 변경. config 16화구·예제·docs 일괄 반영. |
| 2026-06-14 | core/state_machine.py | **EMPTY 전환 시 `_reset_timer()` 호출** — DONE_FIRST/WAIT_SECOND/DONE_SECOND → EMPTY 전환에서 `_pot_absent_count`만 리셋하고 `_done_first_end` 등은 stale로 남던 문제. 세 지점 모두 `_reset_timer()`로 일괄 정리. |
| 2026-06-14 | CLAUDE.md | **문서 7상태 동기화** — 코드는 이미 WAIT_SECOND 포함 7상태인데 문서는 6상태 구버전 유지. 상태표(색상 RGB·진동 활성)·전환 규칙·DONE_FIRST 냉각 단계 설명·config 예시(done_first_timeout/pot_absent_threshold/normalize, rms 0.5→0.20) 갱신. |
| 2026-06-15 | core/state_machine.py | **manual_start의 DONE_FIRST → '바로 재벌'** — 기존 '건너뜀'(WAIT_SECOND 점프)에서 즉시 `_start_second()`(재벌 카운트다운 시작)로 변경. 초벌 완료 후 곧장 이어 조리하는 경우 10분 냉각 대기를 수동으로 건너뛰기 위함. (UI 버튼 라벨 "건너뜀"→"바로 재벌"은 0612UI작업.md로 위임) |
| 2026-06-15 | core/state_machine.py | **seconds_since_done / is_counting property 추가** — UI 유동 정렬(카운트다운 임박순 상단 + 완료 직후 N초 강조)용 조회 인터페이스. 상태 로직 변경 없음. |
| 2026-06-15 | make_sounds.py, assets/sounds/ | **알림음 합성·음량 강화** — 초벌·재벌 공통 2시점(완료 30초 전 / 완료)용 `warn_30s`(2회 비프+상승)·`complete`(4음 팡파레) wav. 주방 소음 대응: 2·3배음 혼합 + sustain 엔벨로프 + 음량 0.9. 외부 음원·인터넷 불필요. 재생 연결은 UI 작업(0612UI작업.md 작업E). |
| 2026-06-15 | 0612UI작업.md | **UI 개선 명세서 작성** — UI 작업을 별도 모델/세션에서 수행하기 위한 단독 문서. 항목 A~F(번호 확대·재벌대기 중복 제거·RMS dev전용·바로재벌 버튼·알림음·유동정렬) + 의존할 백엔드 API/상태 컨텍스트 + 테스트 체크리스트. |
| 2026-06-16 | core/state_machine.py | **6→7상태, WAIT_SECOND 분리** — `DONE_FIRST`(냉각, 진동 무시)와 `WAIT_SECOND`(재벌대기, 진동 재활성)를 별도 상태로 분리. 기존엔 한 상태 안에서 플래그(`_seen_rest_after_first`)로 구분해 딸랑이가 잠깐 멈추면 바로 재벌로 오인할 위험이 있었음. `done_first_timeout`(기본 120초)로 자동 전환. |
| 2026-06-16 | core/state_machine.py | **`pot_absent_threshold` debounce 도입** — DONE_FIRST/WAIT_SECOND/DONE_SECOND에서 pot 미감지 시 즉시 EMPTY가 아니라 연속 N프레임(기본 30, `_body_ttl` 15프레임과 합쳐 총 약 5초) 못 봐야 EMPTY 확정. 손/김에 순간 가려도 타이머 보존. |
| 2026-06-16 | ui/ui_display.py | **UI 전면 재설계 — 좌/우 분리 패널 폐지 → 통합 캔버스** — 카메라 박스+화구 카드를 반칸(half-cell) 그리드 위에 함께 자유 배치. 매장 실제 배치(ㄱ자, 분리 블록 등) 재현 목적. 배치 모드(`L`키)로 드래그 이동(좌측상단 그랩 오프셋 기준, 미리보기=실제 스냅 위치 일치)·카메라 리사이즈(↘ 핸들) 지원. 카드 자동 정렬(urgency sort) 폐지, 저장된 위치 고정 표시로 전환. |
| 2026-06-16 | ui/ui_display.py | **화구 카드 번호 확대** — 화구 번호를 카드에서 가장 크고 눈에 잘 보이는 요소로 변경(카드 높이에 비례하는 동적 폰트, 상단 대부분 차지, 가로 중앙). 상태/타이머는 번호 아래 보조 한 줄로 통합. 폰트 크기별 캐싱(`_dynamic_font`)으로 성능 영향 없음. |
| 2026-06-16 | config/store_config.json | **`normalize_gamma` 0.6 → 1.0 환원** — 화구별 실제 bbox_diag 분포 데이터 없이 정해진 0.6값(거리별 보정 강도 완화)을 근거 부족으로 보류, 기존 검증된 완전 보정(1.0)으로 환원. 코드 자체는 유지 — 추후 `diag_rms.py`로 실측 후 데이터 기반 재조정 예정. |
| 2026-06-16 | config/store_config.json, ui/ui_display.py, main.py | **타이머 값 전체 화구 통일** — 캘리브레이션으로 신규 추가된 화구(5~15번)가 구버전 기본값(countdown_second=300/done_first_timeout=600)을 가져가 1~4번(270/120)과 갈라져 있던 것을 270/120으로 통일. 캘리브레이션 신규화구 기본값·main.py config 폴백값도 동일하게 동기화. |
| 2026-07-08 | sources/video_source.py | **카메라 해상도 고정(1920×1080) + 순차 폴백(1280×720→640×480)** — 드라이버 기본 해상도로 열려 ROI 캘리브레이션이 카메라 교체마다 어긋나던 문제. `isOpened()`만으론 불충분(일부 카메라는 미지원 해상도에도 True 반환하며 read() 실패)해 실제 프레임 read 성공까지 확인 후 채택하도록 구현. |
| 2026-07-09 | core/frame_processor.py, core/state_machine.py, core/data_logger.py, ui/ui_display.py | **딸랑이/vent 오분류로 인한 상태 와리가리 해결** — YOLO의 weight↔vent 오분류 + body 검출 오차로 인한 화구 간 window 침범(예: 13번이 11번의 vent를 채감)이 STEAMING 판정을 불안정하게 만들던 문제. weight+vent 통합 후보 풀에서 면적 최댓값만 채택 + 화구 window를 자기 ROI로 클리핑. 진단용 vent_count/weight_class_count 필드와 dev 오버레이 추가. |
| 2026-07-09 | core/stabilizer.py | **해상도 변경 시 크래시 방지** — 카메라 재연결 등으로 프레임 크기가 바뀌면 `calcOpticalFlowPyrLK`가 이전 프레임과 크기 불일치로 죽던 문제. 이전 프레임과 shape 다르면 첫 프레임과 동일하게 특징점 재검출 + 보정 스킵, `_smooth_dx/dy` 리셋. 2026-07-08의 해상도 폴백 로직과 연동(폴백 시 크기가 바뀌므로). |

---

## Phase 3 실패 기록 (재시도 방지용)

주파수 분석(1~8Hz bandpass)으로 딸랑이 진동 주파수를 추출하려 했으나 전부 실패.

| 시도 | 신호 | 실패 원인 |
|------|------|---------|
| 픽셀 밝기 FFT | bbox 평균 밝기 | 주방 조명이 1~8Hz로 변동 → 오탐 |
| YOLO 중심점 FFT | 중심점 x좌표 | YOLO 정수 좌표 → 0.5px 진폭이 양자화 노이즈에 묻힘 |
| 전체 mean_flow_x FFT | bbox 평균 수평 flow | 배경 픽셀이 방향 신호 희석 → 평균 ≈ 0 |
| 마스크 masked_flow_x | 마스크 픽셀 평균 수평 flow | 마스크 추출률 10~20%, amp 평균 0.054 (임계값 0.3 미달) |

**결론**: Phase 2만으로 충분 (Burner 9 진동 30.7% vs Burner 10 정지 2.7%, 10배 차이).

---

## MaskIoU 폐기 이유 (재도입 방지용)

MaskIoU 방식의 근본 한계:
- YOLO seg 마스크가 프레임당 10~20%만 추출됨 (연기·조명에 따라 누락 심각)
- IoU 신호 자체가 너무 희박해 window 기반 판정이 불안정
- NCC 대비 장점(3D 회전 감지, 연기 FP 제거)이 있었으나 신호 희박 문제가 더 큼
