"""
YOLOv8 학습 스크립트

사용:
    uv run python train.py               # 기본 (yolov8n, imgsz=1280)
    uv run python train.py --model s     # yolov8s 사용 (더 정확, 느림)
    uv run python train.py --imgsz 640   # 메모리 부족 시

학습 클래스:
  0: empty_burner  — 빈 화구
  1: pot_body      — 밥솥 몸체
  2: pot_weight    — 딸랑이 (추)  ← 작은 물체, imgsz 크게 잡는 게 핵심

주요 최적화 포인트:
  - imgsz=640 : Roboflow 640x640 export 기준
  - epochs=150, patience=40 : 충분한 학습 + early stop
  - batch=16 : RTX 3060 Ti 기준 imgsz=640 적정값
  - close_mosaic=20 : 후반 20 epoch에서 mosaic 비활성화 → 안정화
  - copy_paste=0.3 : 작은 물체(딸랑이) 복사·붙여넣기 증강
"""

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO  # type: ignore

DATA_YAML = "dataset/data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="n", choices=["n", "s", "m"],
                        help="YOLOv8 크기: n(빠름) / s(균형) / m(정확)")
    parser.add_argument("--imgsz",  type=int, default=640,
                        help="입력 해상도 (Roboflow 640x640 export 기준)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch",  type=int, default=16,
                        help="RTX 3060 Ti: imgsz=640→16")
    args = parser.parse_args()

    weights = f"yolov8{args.model}-pose.pt"
    print(f"모델: {weights}  |  imgsz: {args.imgsz}  |  epochs: {args.epochs}")

    model = YOLO(weights)

    model.train(
        data    = DATA_YAML,
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = "0" if torch.cuda.is_available() else "cpu",
        project = "runs",
        name    = "pot_detector",
        exist_ok= True,

        # ── Early stopping ───────────────────────────────────────────────
        patience = 8,          # 40 epoch 동안 mAP 개선 없으면 자동 종료

        # ── 소물체(딸랑이) 감지 강화 ──────────────────────────────────────
        copy_paste = 0.3,       # 작은 물체를 다른 배경에 복사·붙여넣기 증강
        mosaic     = 1.0,       # mosaic 증강 (기본값 유지)
        close_mosaic = 20,      # 후반 20 epoch은 mosaic 끔 → 안정화

        # ── 학습률 ───────────────────────────────────────────────────────
        lr0  = 0.001,           # 초기 학습률 (0.01에서 0.001로 낮춤 - 파인튜닝 안정성 강화)
        lrf  = 0.01,            # 최종 학습률 = lr0 * lrf (cosine decay)

        # ── 색상·기하 증강 (주방 환경 대응) ──────────────────────────────
        # ── 색상·기하 증강 (주방 환경 대응) ──────────────────────────────
        hsv_h = 0.015,          # 색조 변화 (기본)
        hsv_s = 0.7,            # 채도 변화 (기본)
        hsv_v = 0.4,            # 밝기 변화 (기본) — 주방 조명 다양성
        degrees   = 0.0,        # 오프라인 증강으로 대체됨
        translate = 0.0,        # 오프라인 증강으로 대체됨
        scale     = 0.0,        # 오프라인 증강으로 대체됨
        fliplr    = 0.0,        # 오프라인 증강으로 대체됨
        erasing   = 0.4,        # 일부 가림 처리 (Cutout / Erasing 유지)
        mixup     = 0.1,        # 이미지 겹치기 (유지)
        
        # ── 기타 ─────────────────────────────────────────────────────────
        val      = True,        # 매 epoch 검증
        save     = True,
        plots    = True,        # 학습 곡선 저장
        verbose  = True,
    )

    best = Path("runs/pose/pot_detector/weights/best.pt")
    if best.exists():
        Path("models").mkdir(exist_ok=True)
        shutil.copy(best, "models/pot_pose.pt")
        print("학습 완료 → models/pot_pose.pt")
    else:
        print("학습 실패. runs/pose/pot_detector/weights/ 확인 필요")


if __name__ == "__main__":
    main()
