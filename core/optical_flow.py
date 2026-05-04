from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np


class OpticalFlowDetector:
    """Dense optical-flow based motion detector for one tracked weight ROI.

    2026-04-19 변경:
    - mask_xy(segmentation 폴리곤) 내부 픽셀만 RMS 계산에 사용 (배경 희석 제거)
    - mask centroid에 EMA 적용 → crop 위치 안정화 (YOLO bbox jitter 완화, upstream)
    - mask 없는 프레임(연기·가림)은 기존 bbox + 전체 RMS 방식으로 fallback
    """

    def __init__(self, flow_cfg: dict) -> None:
        self._enabled = flow_cfg.get("enabled", True)
        self._fb_params = dict(
            pyr_scale=flow_cfg.get("farneback_pyr_scale", 0.5),
            levels=flow_cfg.get("farneback_levels", 3),
            winsize=flow_cfg.get("farneback_winsize", 15),
            iterations=flow_cfg.get("farneback_iterations", 3),
            poly_n=flow_cfg.get("farneback_poly_n", 5),
            poly_sigma=flow_cfg.get("farneback_poly_sigma", 1.2),
            flags=0,
        )

        self._rms_thr          = flow_cfg.get("rms_threshold", 0.5)
        self._rms_ema_alpha    = float(flow_cfg.get("rms_ema_alpha", 0.5))
        self._pos_alpha        = float(flow_cfg.get("pos_ema_alpha", 0.3))  # centroid EMA 계수
        self._window           = flow_cfg.get("window_frames", 25)
        self._trigger          = flow_cfg.get("trigger_frames", 12)
        self._max_box_jump_ratio = float(flow_cfg.get("max_box_jump_ratio", 0.5))
        self._reset_on_jump      = flow_cfg.get("reset_on_box_jump", True)
        self._reset_on_missing = flow_cfg.get("reset_on_missing_box", True)

        self._prev_roi_gray: np.ndarray | None = None
        self._prev_raw_box:  tuple[int, int, int, int] | None = None  # jump 감지 전용
        self._ema_cx: float | None = None  # mask centroid EMA x
        self._ema_cy: float | None = None  # mask centroid EMA y
        self._cv_hist: deque[bool] = deque(maxlen=self._window)
        self._ema_rms: float = 0.0

        self.last_rms:            float = 0.0
        self.last_normalized_rms: float = 0.0
        self.last_smoothed_rms:   float = 0.0
        self.last_flow:           np.ndarray | None = None
        self.last_masked_flow_x:  float | None = None  # Phase 3용
        self.last_roi_box:        tuple[int, int, int, int] | None = None
        self.last_skipped:        bool = False
        self.last_jump_px:        float = 0.0
        self.last_reset_reason:   str | None = None

    def reset(self) -> None:
        self._prev_roi_gray      = None
        self._prev_raw_box       = None
        self._ema_cx             = None
        self._ema_cy             = None
        self._cv_hist.clear()
        self._ema_rms            = 0.0
        self.last_rms            = 0.0
        self.last_normalized_rms = 0.0
        self.last_smoothed_rms   = 0.0
        self.last_flow           = None
        self.last_masked_flow_x  = None
        self.last_roi_box        = None
        self.last_skipped        = False
        self.last_jump_px        = 0.0
        self.last_reset_reason   = None

    @property
    def score(self) -> float:
        if not self._cv_hist:
            return 0.0
        motion_ratio = min(1.0, sum(self._cv_hist) / self._trigger)
        window_ratio = min(1.0, len(self._cv_hist) / self._window)
        return motion_ratio * window_ratio

    @property
    def window_votes(self) -> tuple[int, int, int]:
        """(window 내 motion 프레임 수, 현재 window 길이, trigger 임계값)"""
        return (sum(self._cv_hist), len(self._cv_hist), self._trigger)

    def update(
        self,
        frame:   np.ndarray,
        w_box:   tuple[int, int, int, int] | None,
        mask_xy: np.ndarray | None = None,
    ) -> tuple[bool, float]:
        self.last_skipped      = False
        self.last_jump_px      = 0.0
        self.last_reset_reason = None

        if not self._enabled:
            return False, 0.0

        if w_box is None:
            self._handle_gap("missing_box", clear_history=self._reset_on_missing)
            return self._check(), self.last_rms

        x1, y1, x2, y2 = self._clip_box(frame, w_box)
        if x2 - x1 < 8 or y2 - y1 < 8:
            self._handle_gap("tiny_box", clear_history=self._reset_on_missing)
            return self._check(), self.last_rms

        # ── 1. Box jump 감지 (raw bbox 기준 — 다른 물체로 전환 여부) ──────
        if self._prev_raw_box is not None:
            self.last_jump_px = self._box_jump_px(self._prev_raw_box, (x1, y1, x2, y2))
            bbox_diag = math.hypot(x2 - x1, y2 - y1)
            jump_triggered = bbox_diag > 0 and (self.last_jump_px / bbox_diag) > self._max_box_jump_ratio
            if jump_triggered:
                self.last_skipped  = True
                self._ema_cx = self._ema_cy = None   # 다른 물체 → centroid EMA 리셋
                new_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                self._start_new_track(new_gray, reason="box_jump",
                                      clear_history=self._reset_on_jump)
                self._prev_raw_box = (x1, y1, x2, y2)
                return self._check(), self.last_rms

        self._prev_raw_box = (x1, y1, x2, y2)

        # ── 2. Centroid EMA → 안정된 crop 위치 계산 ─────────────────────
        #   mask 있으면: centroid EMA 갱신 후 안정된 위치로 crop
        #   mask 없으면: raw bbox 그대로 사용 (fallback)
        rx1, ry1, rx2, ry2 = self._get_stable_crop(frame, x1, y1, x2, y2)
        roi_gray = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY)
        self.last_roi_box = (rx1, ry1, rx2, ry2)

        if self._prev_roi_gray is None or self._prev_roi_gray.shape != roi_gray.shape:
            self._start_new_track(roi_gray, reason="new_track", clear_history=False)
            return self._check(), self.last_rms

        # ── 3. Farneback Dense Optical Flow ──────────────────────────────
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_roi_gray, roi_gray, None, **self._fb_params,
        )
        self.last_flow          = flow
        self.last_masked_flow_x = self._compute_masked_flow_x(flow, rx1, ry1, mask_xy)

        # ── 4. RMS 계산: mask 영역 우선, 없으면 bbox 전체 fallback ────────
        magnitude  = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        masked_rms = self._compute_masked_rms(magnitude, rx1, ry1, mask_xy)
        rms        = masked_rms if masked_rms is not None else float(np.sqrt(np.mean(magnitude ** 2)))
        self.last_rms = rms

        # ── 5. RMS EMA 스무딩 (잔여 스파이크 억제) ───────────────────────
        # bbox_size 정규화 제거: 정규화하면 큰 딸랑이일수록 RMS가 작아져 감지 역전 발생
        self.last_normalized_rms = rms  # 호환성 유지 (raw rms 그대로)
        self._ema_rms          = self._rms_ema_alpha * rms + (1.0 - self._rms_ema_alpha) * self._ema_rms
        self.last_smoothed_rms = self._ema_rms

        motion = self._ema_rms > self._rms_thr
        self._cv_hist.append(motion)

        self._prev_roi_gray = roi_gray
        return self._check(), rms

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────

    def _get_stable_crop(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ) -> tuple[int, int, int, int]:
        """bbox center EMA로 안정된 crop 위치 반환.

        항상 bbox center를 EMA에 반영 — mask 유무와 무관하게 일관된 기준.
        mask centroid는 mask 있을 때만 나오므로 crop 기준으로 쓰면
        mask/no-mask 프레임마다 crop이 흔들려 오히려 RMS가 높아짐.

        mask_xy는 RMS 계산(_compute_masked_rms)에만 활용.
        """
        raw_cx = (x1 + x2) / 2.0
        raw_cy = (y1 + y2) / 2.0

        if self._ema_cx is None:
            self._ema_cx, self._ema_cy = raw_cx, raw_cy
        else:
            self._ema_cx = self._pos_alpha * raw_cx + (1.0 - self._pos_alpha) * self._ema_cx
            self._ema_cy = self._pos_alpha * raw_cy + (1.0 - self._pos_alpha) * self._ema_cy

        hw = (x2 - x1) // 2
        hh = (y2 - y1) // 2
        fh, fw = frame.shape[:2]
        sx1 = max(0, int(self._ema_cx) - hw)
        sy1 = max(0, int(self._ema_cy) - hh)
        sx2 = min(fw, sx1 + 2 * hw)
        sy2 = min(fh, sy1 + 2 * hh)
        return (sx1, sy1, sx2, sy2)

    def _handle_gap(self, reason: str, *, clear_history: bool) -> None:
        self.last_reset_reason  = reason
        self.last_rms           = 0.0
        self.last_smoothed_rms  = self._ema_rms  # EMA 값은 유지 (잠깐 가림 대응)
        self.last_flow          = None
        self.last_masked_flow_x = None
        self.last_roi_box       = None
        self._prev_roi_gray     = None
        self._prev_raw_box      = None
        # centroid EMA는 유지 — 잠깐 사라졌다 복귀할 때 위치 안정성 보존
        if clear_history:
            self._cv_hist.clear()
        elif len(self._cv_hist) == self._window:
            self._cv_hist.append(False)

    def _start_new_track(
        self,
        roi_gray: np.ndarray,
        *,
        reason: str,
        clear_history: bool,
    ) -> None:
        self.last_reset_reason  = reason
        self.last_rms           = 0.0
        self.last_smoothed_rms  = self._ema_rms
        self.last_flow          = None
        self.last_masked_flow_x = None
        if reason == "box_jump":
            self._ema_rms          = 0.0
            self.last_smoothed_rms = 0.0
        self._prev_roi_gray = roi_gray
        if clear_history:
            self._cv_hist.clear()

    @staticmethod
    def _compute_masked_rms(
        magnitude: np.ndarray,
        roi_x1: int,
        roi_y1: int,
        mask_xy: np.ndarray | None,
    ) -> float | None:
        """segmentation 폴리곤 내부 픽셀만으로 RMS 계산.
        mask 없거나 유효 픽셀 10개 미만이면 None 반환 → 호출자가 fallback 처리.
        """
        if mask_xy is None or len(mask_xy) < 3:
            return None
        rh, rw = magnitude.shape[:2]
        local_pts = (mask_xy - np.array([[roi_x1, roi_y1]], dtype=np.float32)).astype(np.int32)
        mask_img = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillPoly(mask_img, [local_pts], 255)
        pixel_count = int(np.count_nonzero(mask_img))
        if pixel_count < 10:
            return None
        return float(np.sqrt(np.mean(magnitude[mask_img > 0] ** 2)))

    @staticmethod
    def _compute_masked_flow_x(
        flow: np.ndarray,
        roi_x1: int,
        roi_y1: int,
        mask_xy: np.ndarray | None,
    ) -> float | None:
        """딸랑이 폴리곤 내부 픽셀만 추출해 평균 수평 이동량 반환 (Phase 3용)."""
        if mask_xy is None or len(mask_xy) < 3:
            return None
        rh, rw = flow.shape[:2]
        local_pts = (mask_xy - np.array([[roi_x1, roi_y1]], dtype=np.float32)).astype(np.int32)
        mask_img = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillPoly(mask_img, [local_pts], 255)
        pixel_count = int(np.count_nonzero(mask_img))
        if pixel_count < 10:
            return None
        return float(np.sum(flow[..., 0][mask_img > 0]) / pixel_count)

    def _check(self) -> bool:
        if len(self._cv_hist) < self._window:
            return False
        return sum(self._cv_hist) >= self._trigger

    @staticmethod
    def _clip_box(
        frame: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = box
        return (max(0, x1), max(0, y1), min(fw, x2), min(fh, y2))

    @staticmethod
    def _box_jump_px(
        prev_box: tuple[int, int, int, int],
        curr_box: tuple[int, int, int, int],
    ) -> float:
        px1, py1, px2, py2 = prev_box
        cx1, cy1, cx2, cy2 = curr_box
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        ccx = (cx1 + cx2) / 2.0
        ccy = (cy1 + cy2) / 2.0
        return float(((ccx - pcx) ** 2 + (ccy - pcy) ** 2) ** 0.5)

    @staticmethod
    def flow_to_bgr(flow: np.ndarray) -> np.ndarray:
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
