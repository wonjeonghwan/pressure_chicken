# 매장 시나리오별 config 예제

매장의 카메라 수에 맞는 예제를 골라 복사한 후, ROI 캘리브레이션만 추가하면 됩니다.

## 사용 방법

1. **복사**: 매장에 맞는 예제를 `config/store_config.json`으로 복사
   ```powershell
   copy config\examples\store_4cam.json config\store_config.json
   ```
   또는 `--config` 인자로 직접 지정:
   ```powershell
   uv run python main.py --config config\examples\store_4cam.json
   ```

2. **카메라 연결 확인**: USB 카메라가 모두 인식되어야 합니다. 각 `sources[].index`가 해당 카메라의 OS 인덱스와 일치하는지 확인.

3. **첫 실행 후 캘리브레이션**: `burners`가 비어있으므로 자동으로 캘리브레이션 모드로 진입합니다.
   - `F` 키: 캘리브레이션 모드
   - 카메라가 2대 이상이면 캘리브할 카메라를 먼저 선택 (`F1`~`Fn`)
   - 마우스 드래그로 화구 ROI를 차례로 그리기
   - `Z`: 마지막 ROI 취소 / `ENTER`: 저장 / `ESC`: 취소

## 예제 파일

| 파일 | 카메라 수 | 해상도 | target_fps | 비고 |
|------|----------|--------|-----------|------|
| `store_1cam.json` | 1 | 원본 (≥720p) | 15 | 소형 매장 / 테스트 |
| `store_3cam.json` | 3 | 모두 720p | 10 | source별 override 데모 (어두운 카메라 RMS↑) |
| `store_4cam.json` | 4 | 모두 720p | 10 | **현재 매장 기준** |
| `store_5cam.json` | 5 | 모두 720p | 8 | 대형 매장 (CPU 부하 고려) |

> 카메라 사양은 모두 **720p 이상**으로 통일된 환경을 전제. 480p 강제 다운스케일은 사용하지 않습니다.

## 카메라 모델·환경 혼합 (source별 override)

매장마다 카메라 모델이 다를 수 있고, 한 매장 안에서도 위치별로 조명·각도·거리가 달라 RMS noise floor가 다를 수 있습니다. 그래서 다음 파라미터는 **source별로 덮어쓰기 가능**합니다:

- `resize`, `exposure`, `gamma` — 카메라 화질
- `optical_flow.*` — 감지 민감도 (rms_threshold 등)
- `stabilizer.*` — 흔들림 보정

**합성 우선순위**: `global default < source override < burner override`

예시 (`store_3cam.json`의 2구역) — 어두워서 노이즈가 큰 카메라만 threshold를 더 높임:
```json
{
  "id": 1, "label": "2구역(어두움)",
  "resize": [1280, 720],
  "gamma": 1.3,
  "optical_flow": {
    "rms_threshold": 0.25
  }
}
```
나머지 optical_flow 파라미터는 global을 그대로 사용. **부분 override만 명시**하면 됨.

## 카메라 수 변경

매장에서 카메라를 추가/제거하면:
1. `sources` 배열에서 항목을 추가하거나 제거
2. `target_fps` 조정 (1~2대=15, 3~6대=10, 7대+=8 권장 — 자동 결정 로직은 코드에서 처리)
3. 제거한 카메라를 참조하던 `burners` 항목도 함께 처리 (해당 source_id 가진 화구 제거 또는 다른 카메라로 옮김)
4. 재시작 또는 캘리브레이션 모드(F)에서 핫 리로드

상세 설계: [`MULTI_CAMERA_PLAN.md`](../../MULTI_CAMERA_PLAN.md)
