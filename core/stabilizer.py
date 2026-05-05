"""
영상 안정화 모듈 (Phase 1)

[설계 원칙]
  - goodFeaturesToTrack(Shi-Tomasi 코너) 로 추적 가능한 특징점 동적 선별
    → 텍스처 없는 밋밋한 영역 배제, LK 추적 정확도 향상.
    → 특징점이 부족한 화면(min_inliers 미만) 에서는 균등 Grid 로 폴백.

  - 누적 변환(cumulative) 대신 프레임 간(frame-to-frame) 보정
    → 오차 누적(drift) 없음.
    → EMA(지수이동평균)로 보정량 스무딩 → 급격한 튐 억제.

  - 고주파 떨림만 제거, 저주파 이동은 통과
    → smooth_alpha 로 제어:
       낮은 값(0.1) → 느린 떨림 제거, 빠른 떨림 유지
       높은 값(0.9) → 빠른 떨림만 제거

파라미터 (store_config.json "stabilizer"):
  enabled          : true/false
  grid_rows        : 격자 행 수 (기본 6)
  grid_cols        : 격자 열 수 (기본 8)
  lk_win_size      : LK 윈도우 크기 (기본 21)
  lk_max_level     : LK 피라미드 레벨 (기본 3)
  ransac_threshold : RANSAC 인라이어 판정 거리(px) (기본 3.0)
  smooth_alpha     : EMA 스무딩 계수 0~1 (기본 0.3, 낮을수록 부드럽게)
  min_inliers      : 최소 인라이어 수 (기본 6)
"""

from __future__ import annotations

import cv2
import numpy as np


class Stabilizer:

    def __init__(self, stab_cfg: dict) -> None:
        self._enabled = stab_cfg.get("enabled", True)

        # 격자 설정
        self._grid_rows = stab_cfg.get("grid_rows", 6)
        self._grid_cols = stab_cfg.get("grid_cols", 8)

        # Lucas-Kanade 파라미터
        win = stab_cfg.get("lk_win_size", 21)
        self._lk_params = dict(
            winSize=(win, win),
            maxLevel=stab_cfg.get("lk_max_level", 3),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self._ransac_thr  = stab_cfg.get("ransac_threshold", 3.0)
        self._min_inliers = stab_cfg.get("min_inliers", 6)

        # EMA 스무딩: 보정량을 급격히 적용하지 않고 서서히 적용
        # alpha=1.0: 즉시 적용(스무딩 없음), alpha=0.1: 매우 부드럽게
        self._alpha = stab_cfg.get("smooth_alpha", 0.3)

        self._prev_gray: np.ndarray | None = None
        self._prev_pts:  np.ndarray | None = None  # 이전 프레임 격자점

        # EMA로 스무딩된 현재 보정량 (프레임 간)
        self._smooth_dx: float = 0.0
        self._smooth_dy: float = 0.0

        # 모니터링용 공개 속성
        self.last_n_features: int   = 0
        self.last_n_inliers:  int   = 0
        self.last_raw_dx:     float = 0.0   # RANSAC이 계산한 원본 이동량
        self.last_raw_dy:     float = 0.0
        self.last_smooth_dx:  float = 0.0   # EMA 스무딩 후 적용된 보정량
        self.last_smooth_dy:  float = 0.0

    def reset(self) -> None:
        self._prev_gray  = None
        self._prev_pts   = None
        self._smooth_dx  = 0.0
        self._smooth_dy  = 0.0
        self.last_n_features = 0
        self.last_n_inliers  = 0
        self.last_raw_dx     = 0.0
        self.last_raw_dy     = 0.0
        self.last_smooth_dx  = 0.0
        self.last_smooth_dy  = 0.0

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        if not self._enabled:
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # 첫 프레임: 특징점 생성 후 기준 저장
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_pts  = self._detect_features(gray)
            return frame

        # ── Step 1: LK 광학흐름으로 격자점 추적 ──────────────────────────
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **self._lk_params
        )
        ok        = status.ravel() == 1
        good_prev = self._prev_pts[ok]
        good_curr = curr_pts[ok]
        self.last_n_features = len(good_prev)

        if len(good_prev) < self._min_inliers:
            # 추적 실패 → 특징점 재생성, 보정 없이 반환
            self._prev_gray = gray
            self._prev_pts  = self._detect_features(gray)
            return frame

        # ── Step 2: RANSAC — 배경 인라이어(공통 이동) 선별 ───────────────
        transform, inlier_mask = cv2.estimateAffinePartial2D(
            good_prev, good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=self._ransac_thr,
        )
        self.last_n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

        if transform is None or self.last_n_inliers < self._min_inliers:
            self._prev_gray = gray
            self._prev_pts  = self._detect_features(gray)
            return frame

        # ── Step 3: EMA 스무딩 보정 + warpAffine ─────────────────────────
        raw_dx = float(transform[0, 2])
        raw_dy = float(transform[1, 2])
        self.last_raw_dx = raw_dx
        self.last_raw_dy = raw_dy

        # EMA: 이번 이동량을 서서히 반영 → 급격한 튐 억제
        self._smooth_dx = self._alpha * raw_dx + (1.0 - self._alpha) * self._smooth_dx
        self._smooth_dy = self._alpha * raw_dy + (1.0 - self._alpha) * self._smooth_dy
        self.last_smooth_dx = self._smooth_dx
        self.last_smooth_dy = self._smooth_dy

        # 보정 행렬: 이번 프레임의 이동을 역방향 적용
        correction = np.array([[1, 0, -self._smooth_dx],
                                [0, 1, -self._smooth_dy]], dtype=np.float64)

        stabilized = cv2.warpAffine(
            frame, correction, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # 다음 프레임 준비: raw gray 기준으로 특징점 갱신
        self._prev_gray = gray
        self._prev_pts  = self._detect_features(gray)

        return stabilized

    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        """goodFeaturesToTrack을 사용하여 추적하기 좋은 특징점(코너) 검출. 실패 시 fallback_grid 반환"""
        pts = cv2.goodFeaturesToTrack(
            gray, 
            maxCorners=100, 
            qualityLevel=0.03, 
            minDistance=30, 
            blockSize=7
        )
        if pts is None or len(pts) < self._min_inliers:
            # 밋밋한 화면에서 특징점이 너무 적으면 기존 Grid 방식으로 Fallback
            h, w = gray.shape[:2]
            return self._make_grid(h, w)
        return pts

    def _make_grid(self, h: int, w: int) -> np.ndarray:
        """화면을 grid_rows × grid_cols 균등 격자로 분할, 각 셀 중심점 반환 (Fallback 용)"""
        pts = []
        for r in range(self._grid_rows):
            for c in range(self._grid_cols):
                x = float((c + 0.5) * w / self._grid_cols)
                y = float((r + 0.5) * h / self._grid_rows)
                pts.append([[x, y]])
        return np.array(pts, dtype=np.float32)
