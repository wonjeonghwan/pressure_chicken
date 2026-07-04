"""
YOLOv8 Segmentation 학습 스크립트

사용:
    uv run python train.py               # 기본 (yolov8n-seg, imgsz=960)
    uv run python train.py --model s     # yolov8s-seg 사용 (더 정확, 느림)
    uv run python train.py --imgsz 640 --batch 16   # 메모리 부족 시

학습 클래스 (2026-07 body를 세그멘테이션으로 전환 + vent 신규 추가):
  0: body        — 밥솥 몸체 (기준점, 세그멘테이션)
  1: pot_weight  — 딸랑이 (추)  ← 작은 물체, imgsz 크게 잡는 게 핵심
  2: vent        — 증기 배출구. 실제 파이프라인(core/detector.py)에서는 사용하지
                   않지만, weight와 형태가 비슷해 혼동되는 걸 막기 위해 별도
                   클래스로 학습만 시킴 (미검출 처리는 detector.py 상수 쪽에서)

원본 이미지 해상도는 640x480~4K까지 혼재(Roboflow export 그대로) — imgsz=640
고정은 소물체(딸랑이/vent) 디테일 손실이 크므로 기본값을 960으로 상향.
"""

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO  # type: ignore

DATA_YAML = "dataset/data_aug.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="n", choices=["n", "s", "m"],
                        help="YOLOv8 크기: n(빠름) / s(균형) / m(정확)")
    parser.add_argument("--imgsz",  type=int, default=960,
                        help="입력 해상도 — 원본이 640p~4K로 혼재해 640 고정 시 소물체 손실 큼")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch",  type=int, default=-1,
                        help="-1=AutoBatch(GPU 메모리 프로파일링 후 자동 결정, 기본). "
                             "OOM 반복되면 --batch 4 처럼 직접 고정값 지정")
    args = parser.parse_args()

    weights = f"yolov8{args.model}-seg.pt"
    print(f"모델: {weights}  |  imgsz: {args.imgsz}  |  epochs: {args.epochs}")

    model = YOLO(weights)

    result = model.train(
        data    = DATA_YAML,
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        device  = "0" if torch.cuda.is_available() else "cpu",
        project = "runs",
        name    = "pot_detector",
        exist_ok= True,

        # ── Early stopping ───────────────────────────────────────────────
        # valid가 11장으로 작아(leakage 제거 후) epoch별 val 지표 변동폭이 큼 →
        # 20이면 노이즈로 조기 종료될 수 있어 30으로 완화
        patience = 30,

        # ── 소물체(딸랑이/vent) 감지 강화 ─────────────────────────────────
        copy_paste = 0.0,       # bbox 전용 클래스 혼합 시 마스크 없는 이미지 처리 오류 방지
        mosaic     = 1.0,       # mosaic 증강 (기본값 유지)
        # 데이터가 적어 100 epoch 다 못 가고 patience=30에 조기 종료될 가능성이 큼 →
        # close_mosaic=30이면 마지막 mosaic-off 안정화 구간을 못 밟고 끝날 수 있어 15로 축소
        close_mosaic = 15,
        # mask_ratio는 기본값(4) 유지 — imgsz=960과 함께 1로 낮추면 8GB급 GPU에서 OOM 발생 확인됨.
        # 마스크 정밀도가 꼭 필요하면 imgsz를 낮추고 나서 mask_ratio를 낮추는 식으로 한 번에 하나씩 조정할 것.

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
        workers  = 4,           # CPU 부하 완화 (발열/전력 안정화)
        val      = True,        # 매 epoch 검증
        save     = True,
        plots    = True,        # 학습 곡선 저장
        verbose  = True,
    )

    best = Path(result.save_dir) / "weights/best.pt"
    if best.exists():
        Path("models").mkdir(exist_ok=True)
        dest = Path("models/pot_seg.pt")
        if dest.exists():
            import time
            backup = Path(f"models/pot_seg_prev_{int(time.time())}.pt")
            dest.rename(backup)
            print(f"기존 모델 백업 → {backup.name}")
        shutil.copy(best, dest)
        print(f"학습 완료 → {dest}  (from {best})")
    else:
        print(f"학습 실패. {best} 확인 필요")


if __name__ == "__main__":
    main()
