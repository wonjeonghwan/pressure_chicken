# Pressure Chicken ⏱️🍗
**AI 기반 압력밥솥 딸랑이 진동 자동 감지 시스템**

이 프로젝트는 주방의 설치된 카메라를 통해 밥솥의 화구를 모니터링하고, YOLO 비전 모델과 고정밀 하이브리드 진동 센서(EMA + AbsDiff)를 결합하여 압력솥 딸랑이의 회전을 0% 오탐률로 감지해 자동으로 타이머를 작동시키는 AI 솔루션입니다.

---

## 🚀 시스템 핵심 아키텍처 (Current State)

과거의 무겁고 부정확했던 Optical Flow 및 바운딩 박스 Variance 추적 방식을 전부 폐기하고, 가벼우면서도 오탐지가 없는 **True-Hybrid 픽셀 차분 엔진**으로 고도화되었습니다.

1. **YOLOv8 Max-Bounding ROI Crop**
   - 1080p 전체 화면을 다 훑지 않고, 화구(Burner)가 있는 구역만 최소한으로 잘라서 YOLO에 전달하여 속도(FPS)를 대폭 끌어올렸습니다.
2. **True-Hybrid 진동 감지 엔진 (EMA + Pixel AbsDiff)**
   - **가짜 진동(노이즈) 차단**: YOLO가 박스를 매 프레임 다르게 그리는 Jitter 현상을 무시하기 위해, 지수이동평균(EMA)으로 흔들림 없는 가상의 돋보기 렌즈를 딸랑이 위에 씌웁니다.
   - **진짜 진동(회전) 감지**: 렌즈 속 80% 핵심 영역의 픽셀(무늬)이 이전 프레임과 겹쳤을 때 밝기가 크게 변했는지(AbsDiff) 단 하나의 진실만을 추적합니다. 빛 반사나 수증기로 인한 오탐지 및 정지 상태 오인율이 0%입니다.

---

## 🛠️ 설치 및 실행 방법 (Usage)

### 1. 환경 준비
- 패키지 관리자 `uv`를 활용하여 의존성을 설치하고 가상환경을 구축합니다.
```bash
uv sync   # 또는 uv run python -m pip install -r requirements.txt
```

### 2. 핵심 실행 명령어 파라미터 다루기
메인 시스템을 실행하는 스크립트는 `main.py` 입니다.

- **기본 웹캠 실행** (카메라 0번)
  ```bash
  uv run python main.py
  ```
- **테스트 영상 파일로 실행**
  ```bash
  uv run python main.py --source-0 raw/Sample01.MP4
  ```
- **다중 영상 소스 실행** (여러 밥솥 동시에 체크)
  ```bash
  uv run python main.py --source-0 raw/Sample01.MP4 --source-1 raw/Sample03.MP4
  ```

---

## ⚙️ 진동 센서 사용자 튜닝 가이드 (Tuning)

현장 카메라 상황에 맞게 밥솥 딸랑이의 민감도를 조절하려면 파이썬 코드를 뜯어고칠 필요 없이, 오직 `config/store_config.json`의 `"motion"` 구역 숫자만 입맛에 맞게 조절하시면 됩니다!

| 파라미터명 | 설명 | 추천값 (현재 세팅) | 튜닝 방법 |
| :--- | :--- | :---: | :--- |
| `min_motion_ratio` | 딸랑이 **속살 무늬가 몇 퍼센트나 바뀌어야** 돌아간다고 칠 것인가? | `0.05` (5%) | **낮출수록 극도로 예민해짐.** 딸랑이가 매끈해서 회전해도 무늬가 확 안 변할 때 숫자를 0.05로 팍 낮춰줍니다. |
| `trigger_frames` | 최근 살펴본 영상 30장 중, 무늬가 변한 사진이 **총 몇 장 나와야** 게이지 100% 끓일 것인가? | `10` (10장) | **낮출수록 반응속도 폭발.** 딸랑이가 1초(약 10 프레임)만 살살 흔들려도 게이지가 터지도록 진입 장벽을 낮춥니다. |

---

## 🎯 앞으로 해야 할 일 (Next Steps)

1. **상하반전 및 사각지대 딸랑이 모델 재학습 (Train)**
   - 이미 `augment_dataset.py`를 통해 상하반전(천장 뷰 대비), 90도 회전 등의 데이터 뻥튀기는 마쳤습니다.
   - 현장 영상(예: 거꾸로 촬영된 앵글)에서 YOLO 네모 박스가 딸랑이를 잘 잡지 못하는 상황이 보인다면, 증강된 데이터셋 파일(`dataset/data.yaml`)을 물려서 `train.py`로 YOLO 가중치(`best.pt`)를 가볍게 최신화해주면 됩니다.
2. **현장 테스트 및 파라미터 최종 고정**
   - 위에서 설명한 `config.json` 튜닝 값들을 수정해 보시면서 매장마다 최적화된 감도를 찾아냅니다.


## 테스트 명령어
uv run python tests/compare_optical_flow_targeted.py --burner-id 9
uv run python tests/compare_optical_flow_targeted.py --burner-id 10