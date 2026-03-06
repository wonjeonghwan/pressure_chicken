"""
YOLOv8 학습 스크립트

사용:
    uv run python train.py

학습 완료 후 best.pt 를 models/pot_detector.pt 로 복사.

학습 클래스:
  0: empty_burner  — 빈 화구
  1: pot_body      — 밥솥 몸체
  2: pot_weight    — 딸랑이 (추)

사전 작업:
  1. extract_frames.py 로 프레임 추출
  2. Roboflow 등에서 라벨링 후 YOLOv8 형식으로 export
  3. dataset/ 폴더에 배치
  4. dataset/dataset.yaml 의 클래스 순서 확인
"""

import shutil
from pathlib import Path

import torch
from ultralytics import YOLO  # type: ignore


def main() -> None:
    model = YOLO("yolov8n.pt")  # 최초 실행 시 인터넷에서 자동 다운로드

    results = model.train(
        data    = "dataset/pressure_chicken.v2i.yolov8/data.yaml",
        epochs  = 50,
        imgsz   = 640,
        batch   = 8,        # 메모리 부족 시 4로 줄이기
        device  = "0" if torch.cuda.is_available() else "cpu",
        project = "runs",
        name    = "pot_detector",
        exist_ok= True,
    )

    best = Path("runs/pot_detector/weights/best.pt")
    if best.exists():
        Path("models").mkdir(exist_ok=True)
        shutil.copy(best, "models/pot_detector.pt")
        print("학습 완료. models/pot_detector.pt 저장됨")
    else:
        print("학습 실패. runs/pot_detector/weights/ 확인 필요")


if __name__ == "__main__":
    main()
