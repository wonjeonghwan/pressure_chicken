"""
YOLO 추론 래퍼 — Phase 2

모델 파일이 없으면 자동으로 OpenCV 폴백 모드로 동작하며
model_missing = True 플래그를 설정한다.

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

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2


class BurnerDetector:
    """
    YOLO 기반 감지기. 모델 없으면 OpenCV 폴백.

    YOLO 모델은 전체 인스턴스가 공유한다 (한 번만 로드).
    """

    _model        = None
    _model_missing: bool = True
    _initialized:  bool  = False

    @classmethod
    def _init_model(cls, weights_path: str, confidence: float) -> None:
        if cls._initialized:
            return
        cls._initialized = True

        if not os.path.exists(weights_path):
            print(f"[Detector] 모델 없음: '{weights_path}' → OpenCV 폴백 모드")
            cls._model_missing = True
            return

        try:
            from ultralytics import YOLO  # type: ignore
            cls._model        = YOLO(weights_path)
            cls._confidence   = confidence
            cls._model_missing = False
            print(f"[Detector] YOLO 모델 로드 완료: {weights_path}")
        except Exception as e:
            print(f"[Detector] 로드 실패 ({e}) → OpenCV 폴백 모드")
            cls._model_missing = True

    def __init__(self, weights_path: str, confidence: float = 0.5, motion_cfg: dict | None = None):
        BurnerDetector._init_model(weights_path, confidence)
        self._confidence = confidence
        cfg = motion_cfg or {}

        # OpenCV 폴백용
        self._fgbg       = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self._prev_gray: np.ndarray | None = None
        self._lr         = cfg.get("learning_rate",    0.005)
        self._thresh     = cfg.get("threshold",        15)
        self._mot_ratio  = cfg.get("min_motion_ratio", 0.05)
        self._pot_ratio  = cfg.get("pot_ratio",        0.3)

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

        results = BurnerDetector._model(
            frame,
            conf=self._confidence,
            verbose=False,
        )
        out: list[Detection] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                out.append(Detection(cls_id, conf, x1, y1, x2, y2))
        return out

    def detect_opencv(self, roi_frame: np.ndarray) -> tuple[bool, bool]:
        """
        OpenCV 폴백 감지.
        Returns: (pot_present, motion_detected)
        """
        if roi_frame is None or roi_frame.size == 0:
            return False, False

        roi_area = roi_frame.shape[0] * roi_frame.shape[1]

        # 밥솥 유무 — MOG2
        fgmask = self._fgbg.apply(roi_frame, learningRate=self._lr)
        pot_present = int(np.sum(fgmask > 200)) > int(roi_area * self._pot_ratio)

        # 움직임 — 프레임 diff
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        motion = False
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff   = cv2.absdiff(self._prev_gray, gray)
            motion = int(np.sum(diff > self._thresh)) > int(roi_area * self._mot_ratio)
        self._prev_gray = gray

        return pot_present, motion

    def reset_opencv(self) -> None:
        """ROI 변경 등으로 배경 모델을 초기화할 때 사용"""
        self._fgbg     = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self._prev_gray = None
