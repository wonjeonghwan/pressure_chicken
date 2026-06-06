# 다중 카메라 확장 계획서 (가변 N대)

> 최초 작성: 2026-05-18 | 상태: 설계 단계 (구현 전)
> 단일 카메라 → 임의 N대(1~8+) 동시 모니터링으로 확장하기 위한 분석/설계 문서.
> **카메라 대수는 매장마다, 시점마다 다를 수 있다 (1대로 시작했다가 6대로 늘거나, 4대에서 2대로 줄거나).**

---

## 1. 목표

**한 줄 요약**: 카메라 N대(매장 사정에 따라 1~8+ 가변)를 동시에 운용하며, 카메라 추가/제거가 코드 수정 없이 **config 변경 + 재시작**(중기) / **핫 리로드**(장기)으로 가능해야 한다.

### 충족해야 할 조건
- **가변 카메라 대수**: 1대~8대 이상까지 동일 코드로 동작. 하드코딩된 4 등 매직 넘버 금지.
- **매장별 카메라 수 가변**: A매장 3대 / B매장 4대 / C매장 5대 등 — 동일 코드 + config만으로 대응.
- **카메라 모델/해상도 혼합 허용**: 한 매장 안에서도 source별로 모델·해상도·노출 설정이 다를 수 있음 (예: 메인 라인 1080p 1대 + 보조 480p 3대).
- 카메라 1대 추가/제거 시 사용자가 만지는 것은 **config + (선택적) UI 단축키**뿐
- 저사양 노트북에서 처리 fps **최소 10fps 유지** (카메라 수에 따라 동적 조정 허용)
- 한 카메라가 끊겨도 다른 카메라/화구의 타이머는 영향 없음 (격리)
- 운영자가 한 화면에서 모든 카메라의 화구 상태를 즉시 파악 가능
- 캘리브레이션은 카메라 단위로 독립. 한 대만 재캘리브 가능.

---

## 2. 현재 코드 구조 검토

### 이미 멀티 소스를 전제로 한 부분 (가변 N도 지원)
| 영역 | 코드 위치 | 상태 |
|------|----------|------|
| `sources` 배열 순회 | `main.py:78`, `frame_processor.py:55-65` | **OK** — len(sources) 무관 |
| Source별 VideoSource | `main.py:75-99` | **OK** — id별 dict |
| Source별 Stabilizer | `frame_processor.py:55-59` | **OK** — dict comprehension |
| Source별 YOLO 호출 그룹핑 | `frame_processor.py:120-163` | **OK** (단, 호출은 source별 N회) |
| 화구→Source 매핑 | `burner.source_id` | **OK** |
| BurnerRegistry | `state_machine.py` | **OK** — 카메라 무관 |
| Config 핫 리로드 콜백 | `main.py:156-176` (`handle_config_reloaded`) | **부분 OK** — 화구 변경만 처리, sources 변경은 미지원 |

### 단일 카메라를 가정한 부분 (수정 필요)
| 영역 | 코드 위치 | 문제 |
|------|----------|-----|
| UI 영상 표시 | `ui_display.py:295-299` | `first_cam = list(frames.values())[0]` — 첫 프레임만 표시 |
| 캘리브레이션 ROI source_id | `ui_display.py:267` | `"source_id": 0` 하드코딩 |
| 좌표 변환 `_to_video_pos` | `ui_display.py:211-217` | 단일 `_cam_rect` 기준 |
| 카메라 전환(C키) | `camera_utils.py:switch_camera` | 모든 source 인덱스 일괄 순환 — 다카메라에서 무의미 |
| YOLO 호출 | `frame_processor.py:155, 162` | source별 N회 호출 → batch 통합 가능 |

---

## 3. Config 설계 (가변 대수)

### 3-1. sources 배열 — 카메라 수만큼 추가/제거

```json
"sources": [
  {"id": 0, "type": "camera", "index": 0, "label": "1구역(좌상)", "resize": [640, 480]},
  {"id": 1, "type": "camera", "index": 1, "label": "2구역(우상)", "resize": [640, 480]}
]
```
↑ 2대 운영. 6대로 늘리면 똑같이 4개 항목 더 추가.

**카메라 모델·해상도 혼합 예시** (실제 매장에서 자주 발생):
```json
"sources": [
  {"id": 0, "type": "camera", "index": 0, "label": "메인 라인", "resize": [1280, 720], "exposure": -5},
  {"id": 1, "type": "camera", "index": 1, "label": "보조 라인 1", "resize": [640, 480]},
  {"id": 2, "type": "camera", "index": 2, "label": "보조 라인 2", "resize": [640, 480], "gamma": 1.2},
  {"id": 3, "type": "camera", "index": 3, "label": "외곽", "resize": [640, 480]}
]
```
→ source별로 `resize`/`exposure`/`gamma`를 독립적으로 설정. **각 카메라의 적정 노출/감마는 매장 설치 시 한 번 튜닝해서 config에 박아둠**.

**필드 규약**:
- `id`: 0부터 시작하는 연속 정수. 화구 `source_id`가 가리키는 키.
- `index`: OS 카메라 인덱스 (Windows 한정 — 불안정성 있음, 3-3 참조).
- `label`: UI 표시용. 미지정 시 "카메라 #{id}" 자동 생성.
- `resize`: 강제 다운스케일. 카메라마다 독립. 모델/거리/해상도에 따라 다른 값 가능.
- `exposure`, `gamma`: 카메라 모델·환경별 화질 보정. 독립 설정.
- `type`: "camera" | "file" (테스트용).

**카메라 모델 혼합 시 추가 고려 — Source/Burner Override (구현 완료, 2026-05-19)**:
- `optical_flow`, `stabilizer` 모두 **source별·burner별 부분 override 가능**
- 합성 우선순위: `global default < source override < burner override`
- 사용 예 (`store_3cam.json` 참고):
  ```json
  {
    "id": 1, "label": "2구역(어두움)",
    "resize": [1280, 720], "gamma": 1.3,
    "optical_flow": { "rms_threshold": 0.25 }
  }
  ```
  → 2구역 카메라만 threshold 0.25, 나머지는 global의 0.20 사용
- 구현: [core/frame_processor.py:34-43](core/frame_processor.py#L34-L43) `_merged_cfg()` + 인스턴스 생성 시 합성

**제거 시**: 항목을 빼면 됨. 단, 그 source를 가리키던 화구가 있으면 — 다음 절 참조.

### 3-2. 카메라 추가/제거 시 화구 처리 정책

**시나리오 A**: 카메라 1대 추가 (4대 → 5대)
- `sources`에 `{"id": 4, ...}` 추가
- `burners`에 그 카메라의 화구를 추가 (캘리브레이션 모드에서 F → "카메라 5 선택" → ROI 그리기)
- **재시작 또는 핫 리로드**

**시나리오 B**: 카메라 1대 제거 (4대 → 3대)
- 해당 카메라를 보던 화구들을 어떻게 할지 명시적 결정 필요:
  1. **자동 비활성화**: 해당 burner들을 config에 남기되 `"enabled": false` 플래그 추가 → 카드 회색·"📷없음" 표시
  2. **삭제**: burner 항목 자체 제거 → 카드 사라짐
  3. **다른 카메라로 옮기기**: source_id만 바꿔 다른 카메라에서 보기 (드물지만 카메라 교체 시 유용)
- UI 권장: config 편집 시 "이 카메라를 보던 화구 N개를 어떻게 처리할까요?"라는 선택을 묻는 dialog (Phase C에서 구현). 그 전까진 운영자가 수동 편집.

**시나리오 C**: 카메라가 일시적으로 오프라인 (USB 빠짐, 전원 끊김)
- config 변경 X. 자동 재연결 시도. UI는 "📷오프라인" 표시 + 타이머 잠금 유지.

### 3-3. Windows 카메라 식별 안정성 (재게재)
- 단기: USB 허브 고정 포트 → 재부팅 시 인덱스 안정성 검증
- 중기: `cv2.VideoCapture("video=장치명", cv2.CAP_DSHOW)` 방식 도입 — config의 `index` 대신 `device_name` 옵션
- 장기: USB hardware_id 매칭

### 3-4. defaults 섹션 (선택적)
화구 수가 많아질수록 `countdown_first`, `done_first_timeout` 등이 반복됨. defaults로 빼는 게 합리적이지만 **이번 단계에는 보류**. N≥20개 시 재검토.

---

## 4. UI/UX 설계 — 가변 그리드

### 4-1. 핵심 컨셉: 카메라 수에 따라 레이아웃 자동 결정

`len(sources)` 값으로 그리드를 자동 계산:

| 카메라 수 | 그리드 | 카메라당 화면 비중 |
|----------|--------|--------------------|
| 1대 | 1×1 (단일) | 100% |
| 2대 | 1×2 (가로) | 50% |
| 3~4대 | 2×2 (3대면 1칸 빔 또는 1대 크게+2대 작게) | 25%~50% |
| 5~6대 | 2×3 (가로 3열) | ~17% |
| 7~9대 | 3×3 | ~11% |
| 10대~ | 페이지네이션 (3×3 + 페이지 전환) | ~11% |

**자동 결정 알고리즘** (`ui_display.py`에 추가):
```python
def _grid_layout(n_cams: int) -> tuple[int, int]:
    """n_cams → (rows, cols)"""
    if n_cams <= 1: return (1, 1)
    if n_cams == 2: return (1, 2)
    if n_cams <= 4: return (2, 2)
    if n_cams <= 6: return (2, 3)
    if n_cams <= 9: return (3, 3)
    return (3, 3)  # 10+ → 페이지네이션, 한 페이지 9개
```

> **3대인 경우**: 2×2에서 우하 1칸 빔 → 우측 패널이 그 자리 차지하도록 늘림. 또는 좌측 큰 1대 + 우측 작은 2대 위/아래 분할 옵션. 코드 단순화 위해 **첫 번째 옵션(2×2에서 1칸 빔)** 추천.

### 4-2. 그리드 모드 ↔ 단일 풀뷰 토글 (옵션 D 유지)

**평상시: 자동 그리드**
```
사례 1: 카메라 1대
┌──────────────────────────┬──────────────┐
│       Cam 0 풀뷰          │ 화구 카드    │
└──────────────────────────┴──────────────┘

사례 2: 카메라 2대
┌──────────┬──────────┬──────────────┐
│  Cam 0   │  Cam 1   │  화구 카드   │
└──────────┴──────────┴──────────────┘

사례 3: 카메라 4대
┌─────┬─────┐
│ C0  │ C1  │  ┌──────────────┐
├─────┼─────┤  │  화구 카드   │
│ C2  │ C3  │  │              │
└─────┴─────┘  └──────────────┘

사례 4: 카메라 6대
┌────┬────┬────┐
│ C0 │ C1 │ C2 │ ┌──────────┐
├────┼────┼────┤ │ 화구카드 │
│ C3 │ C4 │ C5 │ │          │
└────┴────┴────┘ └──────────┘
```

**카메라 클릭 시**: 해당 카메라가 풀뷰로 확대. ESC/G 키로 그리드 복귀.

### 4-3. 화구 카드 (가변 환경 대응)

기존 카드에 추가:
- **카메라 배지**: `📷 {label}` 또는 `📷 #{source_id}` — 어느 카메라 소속인지
- **오프라인 표시**: 해당 카메라가 오프라인이면 카드 회색 + `📷❌`
- **현재 풀뷰 카메라 강조**: 단일 풀뷰 모드일 때 해당 카메라 화구만 컬러로, 나머지는 흐림(opacity 30%)

화구 카드의 그리드 컬럼 수도 N과 화구 수에 따라 자동:
- 화구 ≤ 6: 2열
- 화구 7~12: 3열
- 화구 13+: 4열 또는 스크롤

### 4-4. 카메라 단축키 (가변)

```python
# 동적 매핑 (UIDisplay 초기화 시 생성)
self._cam_keys = {
    pygame.K_F1: 0, pygame.K_F2: 1, pygame.K_F3: 2, pygame.K_F4: 3,
    pygame.K_F5: 4, pygame.K_F6: 5, pygame.K_F7: 6, pygame.K_F8: 7,
}
# F{n}: 해당 source_id 풀뷰. 존재하지 않는 source면 무시.
# G    : 그리드 복귀
# ESC  : 풀뷰 → 그리드, 또는 선택 해제
```

| 키 | 동작 |
|----|------|
| `F1`~`F8` | 카메라 1~8 풀뷰 토글 (존재하는 카메라만) |
| `G` | 그리드 모드로 복귀 |
| `ESC` | 풀뷰→그리드, 선택→해제, 캘리브→취소 |
| `1`~`0` | 화구 1~10 선택 (기존 유지) |
| `R` (1초) | 선택 화구 리셋 (기존 유지) |
| `S` | 선택 화구 강제 시작 (기존 유지) |
| `M`, `Space`, `C`, `D`, `Q` | 기존 유지 |
| `F` | 캘리브레이션 진입 (카메라 선택 단계 추가) |

### 4-5. 캘리브레이션 흐름 (카메라 수 무관)

```
F 키 입력
  ↓
[카메라가 1대]  → 바로 캘리브레이션 모드 (현재 동작)
[카메라가 2대+] → 카메라 선택 단계
                  ↓
                  화면에 그리드 표시, "캘리브할 카메라를 클릭하거나 F1~Fn 누르세요"
                  ↓
                  카메라 선택 → 풀뷰 캘리브레이션
                  ↓
                  드래그 → ROI (source_id = 선택된 카메라 id로 자동 부여)
                  ↓
                  C 키: 다른 카메라로 전환 (다른 카메라의 ROI 계속 그리기)
                  ENTER: 저장
                  ESC: 취소
                  Z: 마지막 ROI 취소
```

> 캘리브레이션 중 카메라 전환 가능해야 — 운영자가 한 번에 모든 카메라를 캘리브하는 워크플로우 지원.

**캘리브레이션 중 카메라 전환 시 데이터 보존 정책**:
- `self._calib_burners`는 전체 카메라의 임시 ROI 목록을 통합 보존 (source_id로 구분)
- C 키 또는 F1~Fn 키로 카메라 전환 → 다른 카메라의 ROI는 그대로 유지, 그리기 대상만 전환
- 화면에는 **현재 카메라의 ROI만** 표시 (다른 카메라 ROI는 화면 좌표계가 달라 표시 불가)
- ENTER 저장: 전체 카메라의 ROI를 한 번에 config에 반영
- ESC 취소: 모든 카메라의 변경 사항 폐기 (기존 `burners` 복원)

### 4-6. 화구 ID 부여 규칙

**기본 정책**: 화구 ID는 **글로벌 연속 정수**, 카메라와 무관.
```
카메라 0의 화구: id=1, 2, 3
카메라 1의 화구: id=4, 5
카메라 2의 화구: id=6, 7, 8
```
→ 단축키 `1`~`9`, `0`(10번)으로 직접 선택 가능. ID는 화구 번호이지 카메라 번호가 아님.

**카메라 추가 시**: 새 카메라의 화구는 기존 최대 ID + 1부터 부여 (`max(burner_ids) + 1`).
**카메라 제거 시**: ID는 **재사용하지 않음** (운영자 혼란 방지). 빈 ID는 그대로 남음.
**화구 11+ 케이스**: `1`~`0` 단축키로는 10개까지만 선택. 11번 이상은 마우스 클릭으로만 선택. (다카메라 환경에서는 카드 클릭이 일반적이므로 큰 문제 X.)

> **대안 검토 (보류)**: 화구 ID를 카메라별 prefix로 ("C0-1", "C1-2") — 직관적이나 단축키/state_machine 광범위 변경 필요. 현재 정수 ID 유지 권장.

### 4-7. 카메라 추가/제거 UX (런타임)

**Phase 1 (현재 가능)**: config 파일 편집 후 프로그램 재시작.
**Phase 2 (목표)**: UI에서 카메라 추가 버튼 → "어떤 USB 카메라?" 선택 → config 자동 업데이트 + 핫 리로드.

핫 리로드 확장 (`handle_config_reloaded` 개선):
```python
def handle_config_reloaded(new_cfg):
    # 1. sources 변경 감지: 추가된 source는 새 VideoSource.open(), 제거된 source는 release()
    # 2. burners 변경 감지: 현재 동작 그대로
    # 3. stabilizer/oflow dict 재구성
```

핫 리로드의 위험: STEAMING 진행 중인 화구의 OpticalFlow 히스토리(`_cv_hist`)가 재구성으로 날아갈 수 있음. → 진행 중 화구의 OpticalFlowDetector 인스턴스는 보존하고, 새로 추가된 화구만 새 인스턴스 만드는 식의 **diff-based 재구성** 필요.

### 4-8. DONE_FIRST 잔여 진동 처리 (재게재)

사용자 언급 사항 — `done_first_timeout=600`(10분)으로 잔여 진동을 무시하는 로직 이미 구현됨. 카메라 수가 늘면 동시 DONE_FIRST가 많아질 가능성:

- **카드에 카운트다운 표시**: "초벌완료 07:23" 식으로 남은 잔여 대기시간을 명시 (현재 `status_label` 이미 구현, OK).
- **빠른 종료 UX**: 카드 우클릭 컨텍스트 메뉴(완전 종료/재벌 강제 진입) — 화구 수가 많아질수록 가치 증가.
- **자동 진입 무시**: `S` 키로 즉시 WAIT_SECOND/재벌 점프 가능 (이미 구현).

### 4-9. 알림 (시각 + 청각)

- `DONE_SECOND` 진입: 짧은 비프음 1회
- 카메라 오프라인 5초 이상: 다른 톤 비프음
- pygame.mixer 사용, 매장 환경 고려해 큰 볼륨 + mute 옵션 제공

---

## 5. 성능 최적화 — 카메라 수에 동적 대응

### 5-1. 추정 부하 (카메라 수 N에 따라)

처리 fps target은 N에 반비례 조정 권장:

| N | target fps | 카메라 해상도 | YOLO 호출 | 비고 |
|---|-----------|--------------|-----------|------|
| 1 | 15 | 720p~1080p | 1회/frame | 현재 |
| 2 | 15 | 720p | batch 1회 | 부담 적음 |
| 3~4 | 10 | 720p | batch 1회 | 안전권, 현재 매장 |
| 5~6 | 8~10 | 720p | batch 1회 | 노트북 CPU 한계 임박 |
| 7~9 | 8 | 720p | batch 1회 | CPU 100% 가능성 |
| 10+ | 8 | 720p | batch 1회 | **외부 GPU 권장** |

> 720p 베이스 전제 (사용자 결정, 2026-05-19). 480p 강제 다운은 사용하지 않음.

**자동 조정 로직** (`main.py` 추가):
```python
n_cams = len(sources)
if n_cams <= 2:
    target_fps = 15
elif n_cams <= 6:
    target_fps = 10
else:
    target_fps = 8
# 또는 config에서 명시 가능: "ui.target_fps": 10
```

window/trigger도 fps에 비례 자동 조정 (시간축 1.7초 유지):
```python
# 시간 단위로 표현
trigger_seconds = 1.6   # config로 노출
window_seconds  = 2.5
trigger_frames = round(trigger_seconds * target_fps)
window_frames  = round(window_seconds * target_fps)
```

### 5-2. 최적화 적용 우선순위

1. **카메라 해상도 다운** (`resize`) — N에 따라 자동 조정. 가장 임팩트 큼.
2. **YOLO batch 통합** — `frame_processor.py` 리팩토링. N에 무관하게 1회 호출.
3. **fps 자동 조정** — N에 따라 동적.
4. **UI 영상 캐시** — frame_count 변화 없으면 재사용. 30fps UI 유지하면서 처리 부하 최소화.
5. **goodFeaturesToTrack 빈도 감소** — stabilizer 매 프레임 → 10프레임마다 재검출.
6. **카메라 오프라인 시 해당 source 스킵** — 죽은 카메라에 capture 시도 비용 들이지 않음.

### 5-3. 측정/검증

각 변경마다 측정 (변경 단독 적용 후 정량 비교):
- 평균 frame 처리 시간 (ms)
- CPU/GPU 사용률
- 발열·팬소음 (정성)
- RMS noise floor (`diag_rms.py`) — 해상도 변경 시 재측정 필수

### 5-4. 한계점 명시

- USB 대역폭: USB 3.0 한 포트당 1080p×2~3대 정도. **N≥4면 USB 포트/허브 분산 필수**.
- 노트북 CPU: i5~i7 8세대 기준 N=6, 480p, 10fps가 실용 한계 예상. 실측 후 확정.

### 5-5. 매장 규모별 권장 설정 (운영자용 빠른 참조)

**전제**: 사용 카메라는 모두 **720p 이상**. 480p로 강제 다운하지 않음.

| 매장 규모 | 카메라 수 | 화구 수 | 권장 해상도 | target_fps | 비고 |
|----------|----------|--------|-----------|-----------|------|
| 소형 (1구역) | 1~2대 | 1~6개 | 720p~1080p | 15 | 현재 운영 형태 |
| 중형 (2구역) | 3~4대 | 8~12개 | 720p | 10 | USB 허브 1개, **현재 매장** |
| 대형 (3구역+) | 5~6대 | 13~20개 | 720p | 8~10 | USB 허브 분산, 노트북 CPU 한계권 |
| 초대형 | 7대+ | 20+ | 720p | 8 | **외부 GPU 또는 미니PC 권장** |

config 예제는 별도 `config/examples/` 폴더에 시나리오별로 두는 것을 권장 (Phase A1에서 작성):
- `store_1cam.json`: 단일 카메라 (현재)
- `store_2cam.json`: 2대 카메라
- `store_4cam.json`: 4대 카메라
- `store_6cam.json`: 6대 카메라

---

## 6. 단계별 구현 로드맵

### Phase A: 가변 카메라 capture/연동 (코어)
- A1. `config` 예제 — 1/2/4/6대 시나리오 작성. 동일 코드 동작 확인.
- A2. `FrameProcessor` YOLO batch 통합 리팩토링 — N에 무관 1회 호출.
- A3. `resize`/`target_fps` 자동 결정 로직 (`main.py`).
- A4. 480p, 720p에서 RMS noise floor 재측정 → `rms_threshold` 재조정.
- A5. Windows 카메라 인덱스 안정성 검증 (재부팅, USB 재연결 테스트).

### Phase B: UI 가변 그리드 (옵션 D)
- B1. `_grid_layout(n_cams)` 자동 결정 함수 추가.
- B2. `_draw_camera_area` 가변 그리드 합성으로 변경. source별 `_cam_rect`/`_cam_scale` dict 관리.
- B3. 카메라 클릭 → 단일 풀뷰 토글 (`F1`~`Fn`, `G`, `ESC`).
- B4. 화구 카드 카메라 배지 추가, 카드 그리드 자동 컬럼 결정.
- B5. 캘리브레이션 카메라 선택 단계 추가 (카메라 1대면 스킵).

### Phase C: 카메라 추가/제거 UX
- C1. 카메라 오프라인 감지/표시 (n초 연속 read 실패).
- C2. `handle_config_reloaded` 에 sources 변경 처리 추가 (diff-based 재구성).
- C3. UI에서 카메라 추가/제거 버튼 (선택사항).
- C4. config 편집 시 "이 카메라 화구 N개 처리 방법?" dialog.

### Phase D: 운영 안정성
- D1. 자동 재연결 시도 (USB 잠시 빠졌다 들어왔을 때).
- D2. 카메라 장치명 기반 식별 (Windows DSHOW).
- D3. 알림 사운드 (pygame.mixer).
- D4. 24시간 연속 운용 테스트.

### Phase E: 진단/측정 도구 다카메라 확장
- E1. `diag_rms.py` 멀티 카메라 모드 — 카메라별 RMS noise floor 측정 및 비교
- E2. **신규 `diag_perf.py`** — 카메라 N대 환경에서 다음 지표 측정:
  - 카메라별 capture latency (ms)
  - YOLO 1회 호출 ms (batch 통합 전/후 비교)
  - Stabilizer ms × N
  - OpticalFlow ms × 화구 수
  - 전체 frame 처리 ms / CPU 사용률
- E3. `diag_perf.py` 결과를 csv로 출력 → 매장별 권장 설정 검증/조정 근거

### Phase F: 운영자 친화 설정 UI (Front단 수치 조정)

**목표**: 운영자가 코드/JSON을 직접 만지지 않고 UI에서 주요 파라미터를 슬라이더·숫자입력으로 조정 → 즉시 메모리 반영 + config.json 자동 저장.

**노출할 파라미터 (3계층)**:

| 계층 | 파라미터 | 조정 단위 | UI 위치 |
|------|---------|----------|---------|
| **운영(기본)** | `countdown_first`, `countdown_second` | 화구별 | 화구 카드 더블클릭 → inline 편집 |
| **운영(기본)** | `done_first_timeout` | 매장 전체 (또는 화구별 override) | 설정 드로어 |
| **운영(감지)** | `rms_threshold`, `trigger_frames`, `window_frames` | 매장 전체 (또는 source별) | 설정 드로어 — 실시간 슬라이더 |
| **운영(감지)** | `pot_absent_threshold` | 매장 전체 | 설정 드로어 |
| **설치** | `exposure`, `gamma`, `resize` | source별 | 캘리브레이션 모드 하단 슬라이더 |
| **개발자** | `confidence`, stabilizer/farneback 일체 | 매장 전체 | `D`(Dev) 모드에서만 노출 |

**UI 디자인**:
1. **설정 드로어** — 우측 패널 확장. `O` 키 또는 ⚙ 버튼 클릭 → 영상 위로 슬라이드 인. 슬라이더는 **실시간 적용**(슬라이더 드래그 중에도 즉시 효과). 영상 위에 RMS 수치(`D` 모드)와 함께 보면서 튜닝 가능.
2. **화구 카드 인라인 편집** — 카드 더블클릭 → countdown_first/second만 inline 숫자 편집. 5초 후 자동 저장.
3. **캘리브레이션 모드 하단 슬라이더** — F 키 모드에서 영상 아래에 exposure/gamma 슬라이더 노출. 카메라 화질을 보며 즉시 튜닝.

**저장/적용 방식**:
- 슬라이더 드래그 → 메모리 즉시 반영 (실시간 효과 확인)
- 변경 정지 후 2초 debounce → `config.json` 자동 저장
- 진행 중인 타이머는 보호: `countdown_*` 변경은 다음 사이클부터, `rms_threshold` 등은 즉시
- `R`(reset to default) 버튼 — 매 항목 옆에

**구현 측 고려**:
- `BurnerStateMachine` 의 `_cd_first` 등은 외부에서 변경 가능하도록 setter 또는 직접 속성 접근 허용
- `OpticalFlowDetector._rms_thr` 등도 런타임 변경 가능하게 — 현재 생성자에서 한 번 읽고 끝, **변경 메서드 추가 필요**
- 핫 리로드 콜백(`handle_config_reloaded`)을 슬라이더 변경 시에도 호출 — 단, 매번 OpticalFlow 인스턴스 재생성하면 히스토리 날아감. **부분 update 메서드** 필요

**Phase F 작업 항목**:
- F1. 핵심 모듈에 runtime parameter setter 추가 (`OpticalFlowDetector.set_thresholds`, `BurnerStateMachine.set_countdowns` 등)
- F2. 설정 드로어 UI (`ui_display.py`)
- F3. 화구 카드 인라인 편집
- F4. 캘리브 모드 화질 슬라이더
- F5. config 자동 저장 + debounce
- F6. Dev 모드(`D` 키) 토글로 개발자 파라미터 노출

**우선순위**: Phase F는 **B 이후 D와 병렬** 진행 권장. 운영자가 매장 설치 후 실제 사용하려면 필수.

---

## 7. 위험 / 미결 질문

| 항목 | 내용 | 대응 |
|------|------|------|
| **USB 대역폭** | USB 3.0 한 포트당 1080p×2~3대 한계 | N에 따라 resize 자동 다운, 포트 분산 |
| **카메라 인덱스 불안정** | Windows 재부팅 시 순서 바뀔 가능성 | Phase D2 — 장치명 식별 |
| **noise floor 해상도 의존** | resize 시 RMS scale 변할 수 있음 | Phase A4에서 다중 해상도 측정 |
| **노트북 CPU 한계** | N=6 이상에서 발열·팬 폭발 가능 | target_fps 자동 조정, idle skip |
| **핫 리로드 시 진행 중 타이머 손실 위험** | sources/burners 재구성으로 OpticalFlow 히스토리 날아갈 수 있음 | Phase C2 — diff-based 재구성 |
| **알림 사운드 매장 소음** | 큰 볼륨도 안 들릴 가능성 | 시각 알림 보강(점멸 크게) + 사운드 옵션 |
| **카메라 추가/제거 시 화구 처리** | 제거된 카메라의 화구를 어떻게? | Phase C4 — 명시적 dialog |
| **3대일 때 2×2 빈 칸 처리** | UI 미관 | 빈 칸에 카메라 추가 안내 placeholder |
| **캘리브레이션 중 카메라 전환** | 다른 카메라 ROI 임시 데이터 유지/폐기 정책 | 4-5절 — 통합 임시 버퍼, ENTER 일괄 저장, ESC 일괄 폐기 |
| **화구 ID 재사용 금지로 빈 ID 누적** | 카메라 잦은 추가/제거 시 ID가 띄엄띄엄 | 운영자가 가끔 config 수동 정리. 자동 압축 미지원 |

---

## 8. 다음 액션

### 확정된 운영 조건 (2026-05-19)
- **현재 매장: 4대 운영** → Phase A1의 기준 시나리오는 4대.
- **다른 매장: 3대 / 5대 등 가변** → 코드는 N 대수에 의존하지 않아야 함.
- **카메라 모델·해상도 가변** → source별 독립 설정 필수 (resize/exposure/gamma).

### 실행 순서
1. **Phase A1** (다음): `config/examples/store_4cam.json` 작성 + 4대 capture 검증. 동시에 1대/3대/5대 예제도 작성해 N 가변성 즉시 확인.
2. **Phase A2 (YOLO batch)**: 가장 큰 성능 이득. `frame_processor.py:120-163` 리팩토링.
3. **Phase A3+A4**: `target_fps` 자동 결정 + 480p/720p에서 RMS noise floor 재측정. **카메라 모델이 다르면 source별 측정 필수**.
4. **Phase A5**: Windows 카메라 인덱스 안정성 검증.
5. **Phase B**: UI 가변 그리드 — 핵심 운영자 UX.
6. **Phase C / D / E**: 핫 리로드 / 알림 / 진단 도구.

각 Phase 완료 시 본 문서의 해당 섹션에 결과·조정사항 기록.
