# 압력밥솥 타이머 — 액션플랜

> 최종 업데이트: 2026-05-05 (UI 전면 재설계, RMS 정규화 방식 확정) | 현재 단계: Phase 2 완료 → 현장 라이브 테스트 대기

---

## 현재 진행 위치

```
Phase 0 ✅  →  Phase 1 ✅  →  Phase 2 ✅  →  모델 재학습 ✅  →  현장 테스트 ⏳
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

### Phase 3 — 카메라 2대 확장
- [ ] 2대 동작 확인 (config 설정 완료)
- [ ] 20개 화구 동시 실행 성능 (15fps 유지 목표)

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
