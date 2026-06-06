# 다카메라 UX 개선 + 비용 절감 작업 로그

**날짜**: 2026-06-04
**목표**: 다카메라(4대) 운영 환경에서 UI/UX 개선 + 처리 비용 절감
**범위**: UI 통합 그리드 + 캘리브 멀티카메라 + 카드 뱃지 + 카메라 +/- 버튼 + ROI 면적 계측 + 동적 margin
**제약**: 8월 1일 1차 오픈 사수 / 감지 알고리즘 로직 무수정

---

## 작업 단계

### Phase A — UI/UX 통합 대시보드 개편 (완료)

| Step | 작업 | 결과 |
|------|------|------|
| 1 | 테스트 인프라 (`--headless`, `--screenshot`, 4cam 시뮬 config) | ✅ |
| 2 | 다카메라 균등 그리드 뷰 + 셀 헤더 (`[Cam-N]` + 연결 상태 LED) | ✅ |
| 3 | 캘리브 드래그 → source_id 자동 귀속 | ✅ |
| 4 | 카드 [Cam-N] 뱃지 + 진동 게이지 항상 표시 | ✅ |
| 5 | 캘리브 ROI 클릭 선택 + DEL 삭제 + ID 자동 재정렬 | ✅ |
| 6 | 카메라 +/- 버튼 (우측 패널) + 토스트 메시지 | ✅ |

### Phase B — 매장 배포 준비 (이 문서의 본 작업)

| Step | 작업 | 결과 |
|------|------|------|
| 1 | `.gitignore` 보안 보강 | ✅ `raw/*.xlsx`, `raw/~$*`, `raw/*.csv` 추가. git status에서 거래처 파일 사라짐 확인 |
| 2 | ROI 합집합 면적 계측 로그 | ✅ `FrameProcessor.estimate_roi_coverage()` 추가, main.py에서 시작 시 + 핫리로드 시 출력 |
| 3 | 동적 margin (안전선 30~50) | ✅ `max(30, min(50, ROI_short × 0.1))` 적용. 1cam/4cam 모두 감지 회귀 없음 |
| 4 | `run.bat` 작성 | ✅ uv 자동 체크 + .venv 첫 실행 sync + 인자 전달 가능 |
| 5 | 종합 회귀 테스트 | ✅ 3 시나리오(1cam·4cam+8b·4cam+0b) 전부 에러 없음, 감지 정상 |
| 6 | README + md 최신화 | ✅ README 전면 재작성, ACTION_PLAN/CLAUDE 갱신 |

---

## 상세 결과

### Step 1 — `.gitignore` 보강
- 변경 파일: `.gitignore`
- 추가 항목: `raw/*.xlsx`, `raw/~$*`, `raw/*.csv`
- 동기: 조직 보안 지침 — 거래처/매출/원가 git 금지
- 검증: `git status --short` 실행 시 `raw/구매요청_0513.xlsx` 더 이상 표시 안 됨 ✅

### Step 2 — ROI 합집합 면적 계측 로그
- 변경 파일: `core/frame_processor.py`, `main.py`
- 신규 메서드: `FrameProcessor.estimate_roi_coverage(source_resolution)` — 카메라별 ROI 픽셀 마스크 합집합 / 화면 면적
- main.py에 `print_load_diagnostics()` 헬퍼 추가, 초기화 + 핫 리로드 시 호출
- 출력 예시:
  ```
  [diag] Cam-0: burners=2, ROI 합집합=34.7%  → ✓ 가벼움
  [diag] Cam-1: burners=4, ROI 합집합=78.3%  → ⚠⚠ 카메라 추가 검토
  [diag] Cam-2: 화구 미할당
  ```
- 임계 기준: 40% / 65% (가벼움 / 적정 / 과부담)
- 검증: 4cam config로 4 화구 분담 시 각 34.7% 정상 출력 ✅
- 검증: 1cam 6 burner 시 62.4% → ⚠ 적정 정상 출력 ✅

#### 정적 분석 hint 처리
- 첫 구현에 `for x, y, w, h in rois` 사용 → unused 변수 경고
- `_` 접두사 시도 → 같은 lint 도구가 여전히 잡음
- 최종 해결: `r[0] + r[2]` 형태로 인덱싱 → hint 해소

### Step 3 — 동적 margin
- 변경 파일: `core/frame_processor.py` line 200~232 부근
- 기존: `margin = 50` 고정
- 신규: `margin = max(30, min(50, int(min_short_side * 0.1)))`
- 안전선 근거:
  - 30 미만은 stabilizer warpAffine 후 객체 이탈 위험
  - 50 초과는 작은 ROI 환경에서 YOLO 비용 비대화
- 수식 검증:
  ```
  short_side=100  → margin=30
  short_side=300  → margin=30
  short_side=500  → margin=50
  short_side=800  → margin=50
  ```
- 회귀 검증:
  - 1cam 6 burner: 6개 화구 모두 노란 코너 마크 + #1~#6 라벨 정상 ✅
  - 4cam 8 burner: 8개 화구 전부 매칭 정상 ✅

### Step 4 — run.bat
- 변경 파일: `run.bat` (신규)
- 동작:
  1. 자기 폴더로 `cd`
  2. `uv` 존재 확인 → 없으면 안내 후 종료
  3. `.venv` 없으면 `uv sync` 자동 실행
  4. `uv run python main.py` (또는 인자 전달)
  5. 종료 후 `pause` — 콘솔 유지
- 사용:
  - 더블클릭 → 기본 config로 실행
  - `run.bat --config config/examples/store_4cam.json` → 다른 config 지정

### Step 5 — 종합 회귀 테스트
- 시나리오 1: `config/store_config.json` + `--source-0 raw/Side_01.mov` (1cam 6burner)
  - `target_fps=15.0 (cams=1)` ✅
  - `[diag] Cam-0: burners=6, ROI 합집합=62.4% → ⚠ 적정` ✅
  - 스크린샷: `docs/ux_screenshots/final_1cam.png`
- 시나리오 2: `config/examples/store_4cam_video_with_burners.json` (4cam 8burner)
  - `target_fps=10.0 (cams=4)` ✅
  - 4개 카메라 각 34.7% 출력 ✅
  - 스크린샷: `docs/ux_screenshots/final_4cam_8b.png`
- 시나리오 3: `config/examples/store_4cam_video.json` (4cam 0burner)
  - 화구 미할당 정상 표시 ✅
  - 캘리브 모드 자동 진입 ✅
  - 스크린샷: `docs/ux_screenshots/final_4cam_0b.png`
- 모든 시나리오에서 에러·트레이스백 없음

### Step 6 — README + md 최신화
- `README.md` 전면 재작성:
  - 옛 "True-Hybrid 픽셀 차분 엔진" 설명 폐기 → 실제 Phase 1+2 파이프라인으로 갱신
  - 매장 운영자용 셋업·실행 절차 추가
  - 진단 로그 해석 가이드 추가
  - 단축키 + 우측 패널 UI 안내
  - 자주 발생하는 문제 6가지 표
- `ACTION_PLAN.md` 갱신:
  - 2026-06-04 다카메라 UX + 매장 배포 준비 섹션 추가
  - 현재 진행 위치에 "다카메라 UX ✅" 추가
- `CLAUDE.md` 미세 갱신:
  - YOLO 추론 줄에 "동적 margin" 명시
  - "진단 로그" 한 줄 추가
- 작업 로그(이 파일) 단계별 결과 채움

---

## 비용 절감 효과 (이론)

### YOLO 추론 면적 (margin 영향)
- 기존 (margin=50 고정): ROI 합집합 + 50px × 2축 외곽
- 신규 (동적 margin):
  - 화구 ROI 200×200 → margin 30 → 양축 합 40px 절감
  - 화구 ROI 500×500 → margin 50 → 변화 없음 (큰 ROI 환경에서 안전)
- 작은 ROI가 모여 있는 매장(카메라당 2~4구)에서 약 10~15% YOLO 입력 면적 절감

### 가시성 (계측 로그)
- 매장 셋업 시 어떤 카메라가 부담인지 즉시 확인 가능
- 카메라 추가 결정의 근거 데이터 제공
- 종전: 운영 중 fps 떨어지면 사후 추정 / 신규: 셋업 단계에서 사전 진단

---

## 위험·미해결 사항

| 항목 | 현재 상태 | 향후 |
|------|----------|------|
| 카메라 자동 재연결 | 미구현 (수동 +/- 또는 재시작 필요) | 매장 1차 운영 후 결정 |
| FPS 자동 부하 조절 (화구 수·ROI 면적 반영) | 카메라 수 기반만 동작 | 매장 실측 데이터 보고 종합 점수 추가 |
| 실제 마우스 입력 자동 테스트 | 헤드리스 스크린샷으로 대체 | 별도 작업 (낮은 우선순위) |
| 영상 파일 추가 UI | `+ 추가` 버튼은 실제 카메라만 | 캘리브 모드 외 UI 미정 |

---

---

## 2026-06-04 추가 — 화구 ID 자동 그룹화 정책

### 동기
다카메라 환경에서 그린 순서대로 1,2,3,4... 부여하면 카메라 ↔ ID 매핑이 뒤죽박죽 (예: ID 3이 Cam-2, ID 4가 Cam-1). 운영자가 카드의 [Cam-N] 뱃지를 매번 봐야 함.

### 사용자 결정
- **자동 그룹화** — 저장 시 `source_id` 오름차순 + 같은 카메라 내 list 순서 유지로 자동 정렬
- 정렬 기준: **그린 순서 + 수동 재정렬** (필요 시 캘리브에서 한 칸씩 이동)

### 구현
- `_renumber_calib()` 정렬 로직 — `sorted(key=source_id)` + 1부터 재부여 + 선택 인덱스 재바인딩
- 키바인딩 추가 — `[` / `]` 키로 선택 화구를 같은 카메라 안에서 한 칸 이동 (다른 카메라로는 안 넘어감)
- `save_calibration()` 진입 시 한 번 더 정렬 보장
- 안내 배너에 `[ ]` 키 사용법 표기

### 검증
**단위 테스트** [tests/test_calib_numbering.py](../tests/test_calib_numbering.py) 6개 모두 통과:
- `test_basic_grouping` — Cam-0/1/2/3 순서대로 ID 1~5
- `test_same_camera_order_preserved` — 카메라 내 그린 순서 보존
- `test_move_within_camera_forward` — `]` 키로 한 칸 뒤로 이동
- `test_move_blocked_at_camera_boundary` — 카메라 경계에선 이동 차단
- `test_delete_renumber` — 중간 삭제 후 ID 재정렬
- `test_selected_index_rebinding` — 정렬 후에도 선택 객체 유지

**회귀** 4cam config로 헤드리스 스크린샷 정상 출력 [docs/ux_screenshots/numbering_regress.png](ux_screenshots/numbering_regress.png).

### 발견된 환경 이슈
- Windows PowerShell cp949 콘솔에서 `✓` 출력 시 `UnicodeEncodeError`
- 해결: `uv run python -X utf8 ...` 또는 `set PYTHONUTF8=1` 환경변수
- 매장 노트북에서도 같은 이슈 가능성 있음 — 운영 콘솔이 한글이라면 신경 안 써도 되지만, 디버깅 시 `-X utf8` 권장

---

---

## 2026-06-04 추가 — 안전한 add_camera + 카메라 선택형 제거

### 동기
매장 노트북에서 "카메라 2대 연결했는데 기존 1대 신호 끊김" 현상 발견 (실제 환경 스크린샷 보고). 진단 결과:
- `add_camera()`가 호출한 `VideoSource.find_available_cameras`가 인덱스 0~9를 전부 열고/닫는 과정에서, **이미 메인 프로그램이 잡고 있던 카메라 핸들을 DSHOW 백엔드가 강제 회수**하며 프레임이 끊김
- 또한 `[− 제거]`가 무조건 마지막 카메라만 제거 — 특정 카메라를 골라 제거할 수 없음

### 변경 사항

#### 1. add_camera() 안전 탐색 (`main.py`)
- `VideoSource.find_available_cameras` 호출 제거
- **사용 중이 아닌 인덱스만** 직접 시도, 성공시 등록
- `vs.open()` + 첫 `read()` 둘 다 성공해야 등록 (가상 카메라 필터)
- 이미 열린 카메라 핸들은 **건드리지 않음**

#### 2. 카메라 선택형 제거 (`ui_display.py`, `main.py`)
- `_selected_camera_id` 상태 추가
- 영상 영역 클릭 → 해당 카메라 선택, 셀 외곽에 **노란 4px 스트로크**
- 우측 패널 라벨이 `카메라 4대 (선택: Cam-N)`로 변함
- `[− 제거]` 버튼이 **선택 시 빨강 `- Cam-N`**, 미선택 시 회색 `− 제거`로 표시
- 미선택 상태에서 누르면 토스트: "제거할 카메라를 영상에서 먼저 클릭하세요"
- `remove_camera(source_id)` 콜백 시그니처 변경 (인자 추가)

#### 3. ESC / 캘리브 진입 시 자동 선택 해제
- ESC: `_selected_camera_id = None`
- `start_calibration()`: 카메라 선택 자동 해제 (캘리브 모드는 ROI 선택용)

### 검증
**시뮬레이션 스크립트** [tests/sim_camera_select.py](../tests/sim_camera_select.py) 4 시나리오 통과:
1. Cam-2 셀 클릭 → `_selected_camera_id == 2` + 외곽 스트로크 ✓
2. 같은 셀 다시 클릭 → 선택 해제 ✓
3. Cam-1 선택 → 제거 버튼 클릭 → `remove_camera(1)` 콜백 정상 호출 ✓
4. 선택 없이 제거 버튼 → 토스트 "먼저 클릭하세요" + 콜백 호출 안 됨 ✓

스크린샷:
- [docs/ux_screenshots/cam_select_cam2.png](ux_screenshots/cam_select_cam2.png) — Cam-2 선택 상태
- [docs/ux_screenshots/cam_select_cam3_with_remove_label.png](ux_screenshots/cam_select_cam3_with_remove_label.png) — Cam-3 선택 + 보너스 토스트

### 영향 범위
- 감지 알고리즘 무수정 ✓
- 회귀 테스트 통과 (4cam 8 burner 정상 진단 로그 출력)

---

## 다음 단계 — 매장 노트북 이관 절차

1. `git add . && git commit && git push` (사용자가 직접)
2. 매장 노트북에서:
   - `winget install astral-sh.uv Python.Python.3.11`
   - `git clone <repo>`
   - `cd Pressure_Chicken`
3. 카메라 4대 USB 연결 (좌/우/후면/USB-C 분산)
4. 노트북 절전모드 OFF
5. `run.bat` 더블클릭 → 첫 실행 시 `uv sync` (수 분)
6. F키로 캘리브 → 콘솔 진단 로그 확인 → 모든 카메라 🟢 또는 🟡인지
7. `sources[].exposure` 매장 광량 맞춰 튜닝 (-3 ~ -7)
8. 20~30초 워밍업 후 운영 시작
9. 1~2시간 운영 데이터 수집 → 부하 / fps / false positive 평가
10. 필요 시 FPS 자동 종합 조절 코드 추가

---

## 결정 기록 (Phase B)

### 왜 동적 margin은 미리 넣고, FPS 자동 조절은 보류?

| 항목 | 동적 margin | FPS 자동 조절 |
|---|---|---|
| 적용 시점 | 캘리브 직후 즉시 | 시작 시 / 부하 측정 후 |
| 튜닝 필요? | ❌ ROI 크기에서 자동 계산 | ⚠ 실제 부하 보고 임계값 조정 필요 |
| 효과 검증 | 코드 검토만으로 충분 | 매장 운영 데이터 필요 |
| 위험성 | 안전선만 잡으면 0 | 잘못된 임계값은 fps 폭락 |

→ 동적 margin은 매장 셋업 즉시 효과. FPS는 카메라 수 기반 자동 산정이 이미 존재 ([main.py:63-72](../main.py#L63-L72)), 추가 종합 점수는 매장 실측 후 결정.

### 비용 결정 4축

- **카메라 수**: frame I/O + Stabilizer + YOLO 호출 횟수
- **ROI 합집합 면적**: 각 YOLO 호출당 추론 비용 (단일 최대 레버)
- **화구 수**: Phase 2 호출 횟수
- **딸랑이 bbox 크기**: Phase 2 호출당 Farneback 영역

→ 가장 큰 레버는 **ROI 합집합 면적**. 그래서 계측 + margin 최적화가 효과적.

---
