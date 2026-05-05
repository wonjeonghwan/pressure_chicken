"""
YOLO-seg 추론 래퍼

모델 파일이 없으면 model_missing = True 플래그를 설정하고
detect()는 빈 리스트를 반환한다.

클래스 정의:
  0: empty_burner  — 빈 화구
  1: pot_body      — 밥솥 몸체 (기준점)
  2: pot_weight    — 딸랑이 (추)
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np


CLASS_EMPTY_BURNER = 0
CLASS_POT_BODY     = 1
CLASS_POT_WEIGHT   = 2


@dataclass
class Detection:
    class_id:   int
    confidence: float
    x1: int; y1: int; x2: int; y2: int
    # pose 키포인트 — seg 모델에서는 None
    keypoints: list[tuple[float, float, float]] | None = None
    # seg 폴리곤: shape (N, 2), 이미지 좌표계 — None이면 seg 정보 없음
    mask_xy: np.ndarray | None = None

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2


class BurnerDetector:
    """
    YOLO 기반 감지기. 모델 없으면 빈 리스트 반환.

    YOLO 모델은 전체 인스턴스가 공유한다 (한 번만 로드).
    """

    _model        = None
    _model_missing: bool = True
    _initialized:  bool  = False
    _use_half:     bool  = False  # FP16 사용 여부 (CUDA만 True)

    @classmethod
    def _init_model(cls, weights_path: str, confidence: float) -> None:
        if cls._initialized:
            return
        cls._initialized = True

        if not os.path.exists(weights_path):
            print(f"[Detector] 모델 없음: '{weights_path}' → 빈 리스트 반환 모드")
            cls._model_missing = True
            return

        try:
            import torch
            from ultralytics import YOLO  # type: ignore
            cls._model        = YOLO(weights_path)
            cls._confidence   = confidence
            cls._model_missing = False
            # FP16(half)은 CUDA에서만 안정적 — MPS/CPU는 False
            cls._use_half = torch.cuda.is_available()
            print(f"[Detector] YOLO 모델 로드 완료: {weights_path}  (half={cls._use_half})")
        except Exception as e:
            print(f"[Detector] 로드 실패 ({e}) → 빈 리스트 반환 모드")
            cls._model_missing = True

    def __init__(self, weights_path: str, confidence: float = 0.5):
        BurnerDetector._init_model(weights_path, confidence)
        self._confidence = confidence



    # ── 공개 API ───────────────────────────────────────────────────────

    @property
    def model_missing(self) -> bool:
        return BurnerDetector._model_missing

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        YOLO 추론. frame 전체(또는 ROI 크롭)에 대해 실행.
        모델 없으면 빈 리스트 반환.
        """
        if BurnerDetector._model_missing or frame is None or frame.size == 0:
            return []
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """
        복수 ROI 크롭을 한 번의 YOLO 호출로 처리 (배치 추론).
        화구 N개를 개별 N회 대신 1회로 처리해 추론 비용 대폭 절감.
        Returns: frames 순서에 대응하는 Detection 리스트의 리스트.
        """
        if BurnerDetector._model_missing or not frames:
            return [[] for _ in frames]

        valid_idx = [i for i, f in enumerate(frames) if f is not None and f.size > 0]
        if not valid_idx:
            return [[] for _ in frames]

        valid_frames = [frames[i] for i in valid_idx]
        results = BurnerDetector._model(valid_frames, conf=self._confidence, half=BurnerDetector._use_half, verbose=False)

        out: list[list[Detection]] = [[] for _ in frames]
        for i, r in zip(valid_idx, results):
            for j, box in enumerate(r.boxes):
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 키포인트 추출 (YOLO-pose 모델일 때만 존재)
                kps = None
                if r.keypoints is not None and j < len(r.keypoints.xy):
                    xy      = r.keypoints.xy[j].cpu().numpy()
                    conf_kp = (
                        r.keypoints.conf[j].cpu().numpy()
                        if r.keypoints.conf is not None
                        else np.ones(len(xy))
                    )
                    kps = [(float(xy[k, 0]), float(xy[k, 1]), float(conf_kp[k]))
                           for k in range(len(xy))]

                # 세그멘테이션 마스크 폴리곤 추출 (YOLO-seg 모델일 때만 존재)
                mask_xy = None
                if r.masks is not None and j < len(r.masks.xy):
                    pts = r.masks.xy[j]
                    if len(pts) >= 3:
                        mask_xy = np.array(pts, dtype=np.float32)

                out[i].append(Detection(cls_id, conf, x1, y1, x2, y2, keypoints=kps, mask_xy=mask_xy))
        return out

