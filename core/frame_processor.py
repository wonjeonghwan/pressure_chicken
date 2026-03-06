"""
프레임 처리기

YOLO 모드:
  ROI 크롭 → YOLO 추론 → pot_body / pot_weight 중심 추출
  → 상대좌표(rel_x, rel_y) 계산 → VibrationTracker → 상태머신 갱신

OpenCV 폴백 모드 (모델 없음):
  ROI 크롭 → MOG2 + 프레임 diff → VibrationTracker(OpenCV 모드) → 상태머신 갱신
"""

from __future__ import annotations

from collections import deque

import numpy as np

from sources.video_source import VideoSource
from core.detector import BurnerDetector, CLASS_POT_BODY, CLASS_POT_WEIGHT, CLASS_EMPTY_BURNER
from core.state_machine import BurnerRegistry, BurnerState


# ── 진동 판별기 ─────────────────────────────────────────────────────────────

class VibrationTracker:
    """
    60프레임 슬라이딩 윈도우 기반 딸랑이 진동 판별.

    YOLO 모드:
      - pot_body 고정(std < body_move_limit)
      - pot_weight confidence >= min_confidence 인 프레임 >= trigger_frames
      - rel_x 의 표준편차 >= rel_x_threshold (진동 진폭 확인)

    OpenCV 폴백 모드:
      - motion_detected=True 인 프레임 >= trigger_frames
    """

    def __init__(self, cfg: dict):
        self._window   = cfg.get("window_frames",   60)
        self._trigger  = cfg.get("trigger_frames",  40)
        self._rel_thr  = cfg.get("rel_x_threshold",  3)
        self._body_lim = cfg.get("body_move_limit",  10)
        self._min_conf = cfg.get("min_confidence",   0.5)
        # 시작 직후 MOG2 배경 미학습으로 인한 오탐 방지
        # warmup 동안은 판별 결과를 False 로 고정
        self._warmup   = cfg.get("warmup_frames", 120)
        self._frame_n  = 0

        # YOLO용: (rel_x, body_cx, body_cy, weight_conf)
        self._yolo_hist: deque[tuple] = deque(maxlen=self._window)
        # OpenCV용: True/False
        self._cv_hist: deque[bool] = deque(maxlen=self._window)

    def update_yolo(
        self,
        rel_x: float | None,
        rel_y: float | None,
        body_cx: float | None,
        body_cy: float | None,
        weight_conf: float,
    ) -> bool:
        self._frame_n += 1
        self._yolo_hist.append((rel_x, rel_y, body_cx, body_cy, weight_conf))
        if self._frame_n < self._warmup:
            return False
        return self._check_yolo()

    def update_opencv(self, motion_detected: bool) -> bool:
        self._frame_n += 1
        self._cv_hist.append(motion_detected)
        if self._frame_n < self._warmup:
            return False
        return self._check_opencv()

    def reset(self) -> None:
        self._yolo_hist.clear()
        self._cv_hist.clear()
        self._frame_n = 0

    # ── 내부 판별 ──────────────────────────────────────────────────────

    def _check_yolo(self) -> bool:
        if len(self._yolo_hist) < self._window:
            return False

        # weight 가시 프레임 필터
        valid = [
            (rx, ry, bx, by)
            for rx, ry, bx, by, wc in self._yolo_hist
            if rx is not None and wc >= self._min_conf
        ]
        if len(valid) < self._trigger:
            return False

        # pot_body 안정성 (전체 윈도우 기준)
        body_xs = [bx for _, _, bx, _ in valid if bx is not None]
        body_ys = [by for _, _, _, by in valid if by is not None]
        if not body_xs:
            return False
        if np.std(body_xs) >= self._body_lim or np.std(body_ys) >= self._body_lim:
            return False

        # rel_x 또는 rel_y 중 하나라도 진동 진폭 충족
        # (좌우 진동 + 앞뒤/회전 방향 움직임 모두 커버)
        rel_xs = [rx for rx, _, _, _ in valid]
        rel_ys = [ry for _, ry, _, _ in valid]
        return (float(np.std(rel_xs)) >= self._rel_thr
                or float(np.std(rel_ys)) >= self._rel_thr)

    def _check_opencv(self) -> bool:
        if len(self._cv_hist) < self._window:
            return False
        motion_count = sum(self._cv_hist)
        return motion_count >= self._trigger


# ── 프레임 처리기 ───────────────────────────────────────────────────────────

class FrameProcessor:
    """
    VideoSource 들에서 프레임을 읽고 화구별 감지 → 상태머신 갱신.
    """

    def __init__(
        self,
        sources:      dict[int, VideoSource],
        burner_cfgs:  list[dict],
        registry:     BurnerRegistry,
        detector:     BurnerDetector,
        motion_cfg:   dict,
    ):
        self._sources    = sources
        self._registry   = registry
        self._detector   = detector
        self._burner_map = {b["id"]: b for b in burner_cfgs}

        # 화구별 VibrationTracker
        self._trackers: dict[int, VibrationTracker] = {
            b["id"]: VibrationTracker(motion_cfg)
            for b in burner_cfgs
        }

        # 소스별 최신 프레임 캐시
        self._frame_cache: dict[int, np.ndarray | None] = {}

        # 상태 전환 추적 (DONE_FIRST 진입 시 tracker 리셋용)
        self._prev_states: dict[int, BurnerState] = {
            b["id"]: BurnerState.EMPTY for b in burner_cfgs
        }

    def step(self) -> dict[int, np.ndarray]:
        """
        전체 소스 프레임 읽기 → 화구별 감지 → 상태머신 갱신.
        Returns: {source_id: frame}  (UI 오버레이용)
        """
        # 1) 소스별 프레임 읽기
        frames: dict[int, np.ndarray | None] = {}
        for src_id, src in self._sources.items():
            ret, frame = src.read()
            frames[src_id] = frame if ret else None
        self._frame_cache = frames

        # 2) 화구별 감지 → 상태 갱신
        detections: dict[int, tuple[bool, bool]] = {}
        for bid, cfg in self._burner_map.items():
            src_id = cfg["source_id"]
            frame  = frames.get(src_id)

            if frame is None:
                detections[bid] = (False, False)
                continue

            roi = cfg.get("roi")
            roi_frame = self._crop_roi(frame, roi) if roi else frame

            if self._detector.model_missing:
                pot_present, motion = self._detector.detect_opencv(roi_frame)
                vibrating = self._trackers[bid].update_opencv(motion)
                detections[bid] = (pot_present, vibrating)
            else:
                dets = self._detector.detect(roi_frame)
                pot_present, vibrating = self._process_yolo(bid, dets)
                detections[bid] = (pot_present, vibrating)

        self._registry.update_all(detections)

        # DONE_FIRST 새 진입 화구의 tracker 리셋
        # (초벌 진동 기록이 남아 재벌이 즉시 오탐 트리거되는 것 방지)
        for bid in self._trackers:
            cur = self._registry.get(bid).state
            if (cur == BurnerState.DONE_FIRST
                    and self._prev_states[bid] == BurnerState.POT_STEAMING_FIRST):
                self._trackers[bid].reset()
            self._prev_states[bid] = cur

        return {sid: f for sid, f in frames.items() if f is not None}

    def _process_yolo(
        self,
        burner_id: int,
        dets: list,
    ) -> tuple[bool, bool]:
        """YOLO 감지 결과에서 (pot_present, vibrating) 계산"""

        # empty_burner 만 감지되면 → 확실히 비어있음
        classes = {d.class_id for d in dets}
        if CLASS_EMPTY_BURNER in classes and CLASS_POT_BODY not in classes:
            self._trackers[burner_id].reset()
            return False, False

        # pot_body 찾기 (가장 confidence 높은 것)
        bodies  = [d for d in dets if d.class_id == CLASS_POT_BODY]
        weights = [d for d in dets if d.class_id == CLASS_POT_WEIGHT]

        if not bodies:
            self._trackers[burner_id].update_yolo(None, None, None, None, 0.0)
            return False, False

        body = max(bodies, key=lambda d: d.confidence)
        pot_present = True

        if not weights:
            vibrating = self._trackers[burner_id].update_yolo(
                None, None, float(body.cx), float(body.cy), 0.0
            )
            return pot_present, vibrating

        weight = max(weights, key=lambda d: d.confidence)
        rel_x  = float(weight.cx - body.cx)
        rel_y  = float(weight.cy - body.cy)

        vibrating = self._trackers[burner_id].update_yolo(
            rel_x, rel_y, float(body.cx), float(body.cy), weight.confidence
        )
        return pot_present, vibrating

    @staticmethod
    def _crop_roi(frame: np.ndarray, roi: list[int]) -> np.ndarray:
        """roi: [x, y, w, h]"""
        x, y, w, h = roi
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]
