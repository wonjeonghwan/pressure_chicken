# 압력밥솥 타이머 — 액션플랜

> 최종 업데이트: 2026-06-15 (바로재벌·알림음·백엔드 검증 완료) | 현재 단계: UI 작업(0612UI작업.md) 위임 → 매장 1차 테스트

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
  "normalize_ref_diag": 40.0
}
```
`rms_threshold` 단위: `norm_rms = raw_rms × ref_diag / bbox_diag` 기준 비율.
- 정지 딸랑이 noise floor: norm_rms ≈ 0.015~0.13 (p99 ≈ 0.32)
- threshold 0.20 = noise p90(0.19) 바로 위, window 투표(14/25)로 FP 추가 차단
- `normalize_ref_diag=40.0`: 현재 영상 기준 평균 딸랑이 bbox 대각선(~37px) 기준. 해상도·줌 변경 시 이 값 재측정.

---

## 미결 과제

### 최우선: 현장 카메라 라이브 테스트
- [ ] 딸랑이 움직임 → STEAMING 자동 전환 안정 확인
- [ ] 초벌 완료 → DONE_FIRST → 재벌 딸랑이 재감지 → 재벌 시작 사이클 확인
- [ ] 타이머 잠금 확인 (사람 가림, 연기 발생 시 타이머 유지)
- [ ] 밥솥 이탈 후 재거치 → EMPTY → POT_IDLE 전환 확인

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
